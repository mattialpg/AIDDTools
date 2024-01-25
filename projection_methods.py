import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE, _t_sne
from umap.umap_ import UMAP
import hover_map
import random

import plotly.express as px

def plot_projection(data, method='tsne', labels=None, symbols=None, distance_matrix=None,
                    embedding=False, hover_params=None, jitter=False, **kwarg):

    # data = np.array(data)

    if method == 'tsne':
        data_embedded = TSNE(learning_rate=50, n_iter=5000, **kwarg).fit_transform(data)
    elif method == 'umap':
        data_embedded = UMAP(n_neighbors=10, min_dist=0.1, **kwarg).fit_transform(data)
    df_proj = pd.DataFrame(data_embedded, columns=['COMP_1', 'COMP_2'])

    # if embedding:
    #     n_samples = len(labels)
    #     # Create the initial embedding
    #     X_embedded = 1e-4 * np.random.randn(n_samples, 2).astype(np.float32)
    #     embedding_init = X_embedded.ravel()  # Flatten the two dimensional array to 1D

    #     p = _t_sne._joint_probabilities(distances=distance_matrix,
    #                                     desired_perplexity=perplexity,
    #                                     verbose=False)

    #     # Perform gradient descent
    #     embedding_done = _t_sne._gradient_descent(_t_sne._kl_divergence, embedding_init, 0, n_samples,
    #                                                 kwargs={'P': p, 'degrees_of_freedom': 1,
    #                                                         'n_samples': n_samples, 'n_components': 2})
            
    #     # Get first and second TSNE components into a 2D array
    #     tsne_result = embedding_done[0].reshape(n_samples, 2)
    #     tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1], 'label': labels})

    if jitter:
        stdev = df_proj.std()
        mask_dupl = df_proj.duplicated().tolist()

        if mask_dupl.any():
            new_comps = []
            for row in df_proj[mask_dupl].itertuples():
                new_comp1 = row.COMP_1 + (random.randrange(-100, 100)/100 * stdev.COMP_1)
                new_comp2 = row.COMP_2 + (random.randrange(-100, 100)/100 * stdev.COMP_2)
                new_comps.append([new_comp1, new_comp2])

            df_proj.loc[mask_dupl, ['COMP_1', 'COMP_2']] = new_comps

    if labels.any():
        # Plotly Express assigns data values to discrete colors if the data is non-numeric.
        # If the data is numeric, the color will automatically be considered continuous.
        df_proj['Label'] = labels.astype(str)
        # palette = [x for x in sns.color_palette('husl', n_colors=len(set(labels))).as_hex()]
                        #  color_discrete_map=palette, symbol=symbols)

    if 'display_data' in hover_params.keys():
        df_proj = pd.concat([df_proj, hover_params['display_data']], axis=1)
        data_names = tuple(hover_params['display_data'].columns)

    fig = px.scatter(df_proj, x='COMP_1', y='COMP_2', color='Label', height=700,
                     hover_data=data_names)
    fig.update_layout(template='plotly_white')
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey'), opacity=0.5))

    if 'img_names' in hover_params.keys():
        fig.update_traces(hoverinfo='none', hovertemplate=None)
        hover_map.hover_app(figure=fig, df=df_proj, hover_params=hover_params)
    else:
        fig.show()

    return df_proj