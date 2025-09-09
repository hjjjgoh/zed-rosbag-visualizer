import rerun as rr
import rerun.blueprint as rrb
import numpy as np


# set rerun blueprint
def setup_rerun_blueprint():
    blueprint = rrb.Horizontal(
        rrb.Vertical(
            # 3D 뷰를 탭으로 분리: All / Points
            rrb.Tabs(
                rrb.Spatial3DView(
                    name="3D (All)",
                    origin="world",              # world 아래 모든 엔티티 표시
                    contents= "world/points",
                ),
                rrb.Spatial3DView(
                    name="3D (Points)",
                    origin="world",
                    contents="world/points",     # 이 뷰에선 points만 표시
                ),
            ),
            rrb.TextDocumentView(name="Description", origin="/description"),
            row_shares=[7, 3],
        ),
        rrb.Vertical(
            rrb.Spatial2DView(
                name="RGB & Depth",
                origin="world/camera/image",
                overrides={"world/camera/image/rgb": rr.Image.from_fields(opacity=0.5)},
            ),
            rrb.Tabs(
                rrb.Spatial2DView(
                    name="RGB",
                    origin="world/camera/image",
                    contents="world/camera/image/rgb",
                ),
                rrb.Spatial2DView(
                    name="Depth",
                    origin="world/camera/image",
                    contents="world/camera/image/depth",
                ),
            ),
            name="2D Views",
            row_shares=[3, 3, 2],
        ),
        column_shares=[3, 1],
    )
    rr.send_blueprint(blueprint)
    rr.log("world", rr.ViewCoordinates.RDF, static=True)
    # R: Right, D: Down, F: Forward / [meter]
   

# log description to rerun
def log_description():
    description = """
    It visualizes RGB and Depth data from ZED camera's ROSBAG.  
    - **3D View**: point cloud and camera position  
    - **RGB & Depth**: RGB and Depth information displayed as an overlay  
    - **RGB Tab**: RGB image only  
    - **Depth Tab**: Depth map only  
    """
    rr.log("description", rr.TextDocument(description, media_type=rr.MediaType.MARKDOWN), static=True)
