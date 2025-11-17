import os, re, yaml
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from bioio import BioImage
from scipy.optimize import curve_fit
from scipy.ndimage import laplace
from tqdm import tqdm

CHUNK_SIZE_IN_BYTES = 1000000000
CHUNK_OVERLAP_SIZE = 50

def get_metadata_from_individual_images_in_czi_file(filepath):
    df = pd.DataFrame()
    file_size = os.path.getsize(filepath)

    with open(filepath, 'rb') as f:
        chunk = f.read(CHUNK_SIZE_IN_BYTES)
        chunk_overlap = f.read(CHUNK_OVERLAP_SIZE)

        datetime_binary = []
        focusposition_binary = []
        stagexposition_binary = []
        stageyposition_binary = []

        with tqdm(total=file_size, unit='B', unit_scale=True, desc='Processing') as pbar:
            while True:
                if chunk == b'':
                    break

                datetime_binary.extend(re.findall(rb'<AcquisitionTime>(\d{4}-[01]\d-[0-3]\dT[0-2]\d:[0-5]\d:[0-5]\d\.\d+Z)', chunk))
                focusposition_binary.extend(re.findall(rb'<FocusPosition>([+\-]?\d+\.?\d*)</Focus', chunk))
                stagexposition_binary.extend(re.findall(rb'<StageXPosition>([+\-]?\d+\.?\d*)</StageX', chunk))
                stageyposition_binary.extend(re.findall(rb'<StageYPosition>([+\-]?\d+\.?\d*)</StageY', chunk))

                bytes_read = len(chunk)
                pbar.update(bytes_read)

                chunk = chunk_overlap + f.read(CHUNK_SIZE_IN_BYTES)
                chunk_overlap = f.read(CHUNK_OVERLAP_SIZE)
                chunk += chunk_overlap

            df['AcquisitionTime'] = pd.to_datetime([(dt.decode()) for dt in datetime_binary], utc=True)
            df['FocusPosition'] = [(float(fp.decode())) for fp in focusposition_binary]
            df['StageXPosition'] = [(float(sxp.decode())) for sxp in stagexposition_binary]
            df['StageYPosition'] = [(float(syp.decode())) for syp in stageyposition_binary]

    return df

def get_measurement_from_czi_file(filepath, df):
    czi_data = BioImage(filepath)

    well_labels = []
    average_values = []
    standard_deviations = []
    focus_scores = []
    
    number_of_channels = czi_data.shape[czi_data.dims.order.find('C')]
    number_of_z_slices = czi_data.shape[czi_data.dims.order.find('Z')]

    for scene in czi_data.scenes:
        well_labels.extend(number_of_channels * number_of_z_slices * [scene.split('-')[-1]])

        czi_data.set_scene(scene)
        img = czi_data.get_image_dask_data('YX', c=1, t=1, z=1).compute()

        average_values.extend([float(img.mean())] * number_of_channels * number_of_z_slices)
        standard_deviations.extend([float(img.std())] * number_of_channels * number_of_z_slices)
        focus_scores.extend([laplace(img).var()] * number_of_channels * number_of_z_slices)

    df['Well'] = well_labels
    df['AverageValue'] = average_values
    df['StandardDeviation'] = standard_deviations
    df['FocusScore'] = focus_scores

    return df

def process_metadata(df):
    df['Row'] = df['Well'].str[0]
    df['Column'] = df['Well'].str[1:].astype(int)
    df['RelativeAcquisitionTimeInSeconds'] = (df['AcquisitionTime'] - df['AcquisitionTime'].iloc[0]).dt.total_seconds()

    return df

def quality_control(filepath):
    df_filepath = filepath.replace('.czi', '.json')

    # Individual images metadata
    if os.path.exists(df_filepath):
        individual_metadata = pd.read_json(df_filepath, convert_dates=['AcquisitionTime'])
    else:
        individual_metadata = get_metadata_from_individual_images_in_czi_file(czi_filepath)
        individual_metadata.to_json(df_filepath, indent=4)

    # Measurments from individual images
    if 'Well' not in individual_metadata.columns:
        individual_metadata = get_measurement_from_czi_file(czi_filepath, individual_metadata)
        individual_metadata.to_json(df_filepath, indent=4)

    # Enhancement of metadata (mostly for visualization)
    if 'Row' not in individual_metadata.columns:
        individual_metadata = process_metadata(individual_metadata)
        individual_metadata.to_json(df_filepath, indent=4)

    popt, pcov = curve_fit(saddle, (individual_metadata['StageXPosition'], individual_metadata['StageYPosition']), individual_metadata['FocusPosition'])

    # Visualization
    # fig = go.Figure(data=go.Scatter(
    #     x=individual_metadata['RelativeAcquisitionTimeInSeconds'],
    #     y=individual_metadata['AverageValue']
    # ))
    # fig.show()

    # fig = go.Figure(data=go.Scatter(
    #     x=individual_metadata['Well'],
    #     y=individual_metadata['FocusScore']
    # ))
    # fig.show()
    
    fig = go.Figure(data=[go.Scatter3d(
        x=individual_metadata['StageXPosition'],
        y=individual_metadata['StageYPosition'],
        z=individual_metadata['FocusPosition'],
        mode='markers',
        marker=dict(size=3, color=individual_metadata['FocusPosition'], colorscale='Pinkyl')
    )])
    fig.add_trace(go.Surface(
        x=np.linspace(individual_metadata['StageXPosition'].min(), individual_metadata['StageXPosition'].max(), 100),
        y=np.linspace(individual_metadata['StageYPosition'].min(), individual_metadata['StageYPosition'].max(), 100),
        z=saddle(np.meshgrid(
            np.linspace(individual_metadata['StageXPosition'].min(), individual_metadata['StageXPosition'].max(), 100),
            np.linspace(individual_metadata['StageYPosition'].min(), individual_metadata['StageYPosition'].max(), 100)
        ), *popt).reshape(100, 100),
        opacity=0.3,
        colorscale='Pinkyl',
        showscale=False
    ))
    fig.update_layout(
        title='Original Focus Positions and Fitted Plane',
        template='plotly_dark',
        scene = dict(
            xaxis = dict(title='Stage X Position (um)'),
            yaxis = dict(title='Stage Y Position (um)'),
            zaxis = dict(title='Focus Position (um)')
        ),
    )
    x_range = individual_metadata['StageXPosition'].max()-individual_metadata['StageXPosition'].min()
    y_range = individual_metadata['StageYPosition'].max()-individual_metadata['StageYPosition'].min()
    z_range = individual_metadata['FocusPosition'].max()-individual_metadata['FocusPosition'].min()

    fig['layout']['scene']['aspectmode'] = 'manual'
    fig['layout']['scene']['aspectratio'] = dict(x=1, y=y_range/x_range, z=100*z_range/x_range)
    fig.show()
    fig.write_html(filepath.replace('.czi', '_interactive-original.html'))

    fig = go.Figure(data=[go.Scatter3d(
        x=individual_metadata['StageXPosition'],
        y=individual_metadata['StageYPosition'],
        z=individual_metadata['FocusPosition'] - saddle((individual_metadata['StageXPosition'], individual_metadata['StageYPosition']), *popt),
        mode='markers',
        marker=dict(size=3, color=individual_metadata['FocusPosition'], colorscale='Pinkyl')
    )])
    fig.update_layout(
        title='Focus Position Residuals After Removing Fitted Plane',
        template='plotly_dark',
        scene = dict(
            xaxis = dict(title='Stage X Position (um)'),
            yaxis = dict(title='Stage Y Position (um)'),
            zaxis = dict(title='Focus Position (um)')
        ),
    )
    fig['layout']['scene']['aspectmode'] = 'manual'
    fig['layout']['scene']['aspectratio'] = dict(x=1, y=y_range/x_range, z=100*z_range/x_range)
    fig.show()
    fig.write_html(filepath.replace('.czi', '_interactive-plane-corrected.html'))

def plane(xy, a, b, c, d):
    x, y = xy
    return (-a * x - b * y - d) / c

def saddle(xy, a, b, c, d, e, g, h, i, j):
    x, y = xy
    return a*x**2*y**2 + b*x**2*y + c*x*y**2 + d*x**2 + e*y**2 + g*x*y + h*x + i*y + j

if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    czi_filepath = config['czi_filepath']
    quality_control(czi_filepath)