import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import cv2
import os


def index_to_time(index, frame_rate=30):

    total_seconds = index // frame_rate
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    
    return f"{minutes}:{seconds:02d}"

def plot_3d_umap(dbscan_labels, behavior_umap, video_frames_sample, model_folder, sample_size, eps, min_samples, embedding_frame_name):
    unique_labels = np.unique(dbscan_labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    traces = []
    for label, color in zip(unique_labels, colors):
        ### get points with the current label ###
        cluster_points = behavior_umap[dbscan_labels == label]
        video_frame_text = video_frames_sample[dbscan_labels == label]

        # print(np.shape(cluster_points))
        # print(np.shape(video_frame_text))
        
        ### add trace for the current label ### 
        traces.append(go.Scatter3d(
            x=cluster_points[:, 0],
            y=cluster_points[:, 1],
            z=cluster_points[:, 2],
            mode='markers',
            name=f'Cluster {label}' if label != -1 else 'Noise',
            marker=dict(
                size=3,
                color=f'rgb({color[0]*255},{color[1]*255},{color[2]*255})',
                opacity=0.8
            ),
            text=[f'{vf[0]} {vf[1]} {index_to_time(vf[1])}' for vf in video_frame_text],
        ))

    layout = go.Layout(
        scene=dict(
            xaxis=dict(title='UMAP 1'),
            yaxis=dict(title='UMAP 2'),
            zaxis=dict(title='UMAP 3')
        ),
        margin=dict(l=0, r=0, b=0, t=0),  
        legend=dict(title='DBSCAN Clusters')
    )

    fig = go.Figure(data=traces, layout=layout)
    fig.write_html(os.path.join(model_folder,f"{sample_size}",f"{embedding_frame_name}_dbscan_{eps}_{min_samples}.html"))
    fig.show()

def save_gifs(model_folder, eps, min_samples, video_frames_sample, dbscan_labels, mp4_folder, sample_size, embedding_frame_name):

    output_dir = os.path.join(model_folder, f"{sample_size}", f"{embedding_frame_name}_output_gifs_{eps}_{min_samples}")
    os.makedirs(output_dir, exist_ok=True)

    ### get all of the unique video names ###
    unique_video_names = np.unique([vf[0] for vf in video_frames_sample])

    ### get all of the unique dbscan labels ###
    unique_labels = np.unique(dbscan_labels)

    for video_name in unique_video_names:
        ### get the indices that correspond to the current video name ###
        relevant_indices = [i for i, vf in enumerate(video_frames_sample) if vf[0] == video_name]

        ### extract the video file name from the video_name variable ###
        video_base_name = video_name.split('DLC')[0]
        start_frame = video_name.split('_')[-2]
        end_frame = video_name.split('_')[-1]

        new_video_name = video_base_name + '_' + start_frame + '_' + end_frame

        ### construct the full path to the video file ###
        mp4_path = os.path.join(mp4_folder, new_video_name + '.mp4')

        ### get the mp4 ###
        cap = cv2.VideoCapture(mp4_path)
        if not cap.isOpened():
            print(f"Error opening video file {mp4_path}")
            continue

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        ### for each label in all of the dbscan labels ###
        for label in unique_labels:

            ### make a directory for it ###
            label_dir = os.path.join(output_dir, f"cluster_{label}")
            os.makedirs(label_dir, exist_ok=True)

            ### search relevant indeces (the indeces that contain the video) for dbscan labels that match the current label ###
            label_indices = [i for i in relevant_indices if dbscan_labels[i] == label]
            #print(label_indices)
            
            #### for each indeces that matches both the video and the dbscan label ###
            for i in label_indices:

                vf = video_frames_sample[i]
                frame_index = int(vf[1])

                ### get 0.5 sec before and after ###
                frame_start = max(frame_index - 14, 0)
                frame_end = min(frame_index + 15, 3599)

                ### construct the output path for the gif ###
                output_video_path = os.path.join(label_dir, f"{new_video_name}_frame_{frame_index}.mp4")
                out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
                
                ### read the first frame ###
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
                
                ### read and write each frame within the sec range ###
                for j in range(frame_start, frame_end + 1):
                    ret, frame = cap.read()
                    if not ret:
                        print(f"Error reading frame {j} from {new_video_name}")
                
                    out.write(frame)
                ### release the VideoWriter for this segment ###
                out.release()  
        ### release the video capture object after processing the current video ###
        cap.release()  