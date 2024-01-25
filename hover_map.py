import io, glob
from base64 import b64encode
from dash import Dash, dcc, html, Input, Output, no_update
from PIL import Image
from jupyter_dash import JupyterDash

def hover_app(figure, df, hover_params):

    def image_to_base64(filename):
        im = Image.open(filename)
        buffer = io.BytesIO()
        im.save(buffer, format="jpeg")
        encoded_image = b64encode(buffer.getvalue()).decode()
        im_url = "data:image/jpeg;base64, " + encoded_image
        return im_url

    # app = Dash(__name__)
    app = JupyterDash(__name__)

    app.layout = html.Div(
        className="container",
        children=[
            dcc.Graph(id="graph-2-dcc", figure=figure, responsive=True,
                      style={'width': '90vw', 'height': '90vh'},
                      clear_on_unhover=True),
            dcc.Tooltip(id="graph-tooltip-2", direction='bottom'),
            html.A(html.Button("Download as HTML"), id="download"),
            dcc.Download(id='download_1')
        ])

    @app.callback(
        Output("graph-tooltip-2", "show"),
        Output("graph-tooltip-2", "bbox"),
        Output("graph-tooltip-2", "children"),
        Output("graph-tooltip-2", "direction"),
        Input("graph-2-dcc", "hoverData"))
        
    def display_hover(hoverData):
        if hoverData is None:
            return False, no_update, no_update, no_update

        pt = hoverData["points"][0]
        bbox = pt["bbox"]

        # Select point by xy values
        # pt['pointNumber'] does not work with multiple curves or groups
        x = pt['x']; y = pt['y']
        idx = df.index[(df.COMP_1 == x) & (df.COMP_2 == y)]
        img_name = str(hover_params['img_names'][idx][0])
        logCMC = str(round(hover_params['display_data'][idx][0], 2))

        img = f"FIGS/{img_name}.png"   # <---------------------- the image
        direction = "bottom" if pt["y"] > 1.5 else "top"    # control the position of the tooltip

        # im_matrix = './FRAGMENTS/4.png'
        im_url = image_to_base64(img)
        children = [
            html.Div([
                html.Img(
                    src=im_url,
                    style={"width": "180px", 'display': 'block', 'margin': '0 auto'},
                ),
                html.P(f"{img_name}"),
                html.P(f"logCMC = {logCMC}")
            ])
            ]
        figure.write_html("plotly_graph.html")

        return True, bbox, children, direction
    
    @app.callback(
    Output('download_1','data'),
    Input('download','n_clicks'),prevent_initial_call=True)
    def download_html(n):
        return dcc.send_file("plotly_graph.html")
    
    # app.run_server()
    app.run_server(mode='inline')