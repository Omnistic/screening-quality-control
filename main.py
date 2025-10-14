import os, re
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from bioio import BioImage
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
    
    number_of_channels = czi_data.shape[czi_data.dims.order.find('C')]
    number_of_z_slices = czi_data.shape[czi_data.dims.order.find('Z')]

    for scene in czi_data.scenes:
        well_labels.extend(number_of_channels * number_of_z_slices * [scene.split('-')[-1]])

        czi_data.set_scene(scene)
        img = czi_data.get_image_dask_data('YX', c=1, t=1, z=1).compute()

        average_values.extend([float(img.mean())] * number_of_channels * number_of_z_slices)
        standard_deviations.extend([float(img.std())] * number_of_channels * number_of_z_slices)

    df['Well'] = well_labels
    df['AverageValue'] = average_values
    df['StandardDeviation'] = standard_deviations

    return df

def process_metadata(df):
    df['Row'] = df['Well'].str[0]
    df['Column'] = df['Well'].str[1:].astype(int)
    df['RelativeAcquisitionTimeInSeconds'] = (df['AcquisitionTime'] - df['AcquisitionTime'].iloc[0]).dt.total_seconds()

    return df

if __name__ == '__main__':
    czi_filepath = r'your_file.czi'
    df_filepath = czi_filepath.replace('.czi', '.json')

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

    # Visualization
    fig = go.Figure(data=[go.Scatter3d(
        x=individual_metadata['Column'],
        y=individual_metadata['Row'],
        z=individual_metadata['FocusPosition'],
        mode='markers',
        marker=dict(size=3, color=individual_metadata['FocusPosition'], colorscale='Pinkyl')
    )])
    fig.update_layout(
        template='plotly_dark',
        scene = dict(
            xaxis = dict(
                title='Column',
                tickmode='array',
                tickvals=sorted(individual_metadata['Column'].unique())
            ),
            yaxis = dict(
                title='Row',
                tickmode='array',
                tickvals=sorted(individual_metadata['Row'].unique())
            ),
            zaxis = dict(title='Focus Position (um)')
        ),
    )
    fig.show()
    fig.write_html('surface.html')