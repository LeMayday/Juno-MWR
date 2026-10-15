# modules
import numpy as np
import plotly.graph_objects as go

# local files
from jmag_helper import RJ_polar_ratio


def create_3d_time_viewer(
    A,
    B,
    C,
    D,
    title,
    frame_duration_ms=100,
    output_html=None,
):
    """
    Create an interactive 3D time-series visualization.

    Parameters
    ----------
    A : array_like, shape (N, 3)
        Position at each time step.
    B : array_like, shape (N, 3)
        Unit vector at each time step. The vector is drawn from A[t]
        to A[t] + B[t].
    C : array_like, shape (N, M, 3)
        Candidate points at each time step.
    D : array_like, shape (N, M)
        Boolean mask. C[t, j] is displayed only when D[t, j] is True.
    title : str
        Title for viewer
    frame_duration_ms : int, optional
        Duration of each frame during playback.
    output_html : str or None, optional
        If provided, save the visualization as a standalone HTML file.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The interactive Plotly figure.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)
    D = np.asarray(D, dtype=bool)

    # Validate array shapes.
    if A.ndim != 2 or A.shape[1] != 3:
        raise ValueError(f"A must have shape (N, 3); got {A.shape}.")
    if B.shape != A.shape:
        raise ValueError(f"B must have shape {A.shape}; got {B.shape}.")
    if C.ndim != 3 or C.shape[0] != A.shape[0] or C.shape[2] != 3:
        raise ValueError(
            f"C must have shape (N, M, 3), with N={A.shape[0]}; got {C.shape}."
        )
    if D.shape != C.shape[:2]:
        raise ValueError(f"D must have shape {C.shape[:2]}; got {D.shape}.")

    N = A.shape[0]

    if N == 0:
        raise ValueError("The arrays must contain at least one time step.")

    # ------------------------------------------------------------------
    # Create the static unit sphere.
    # ------------------------------------------------------------------
    theta = np.linspace(0, 2 * np.pi, 80)
    phi = np.linspace(0, np.pi, 40)
    theta_grid, phi_grid = np.meshgrid(theta, phi)

    sphere_x = np.sin(phi_grid) * np.cos(theta_grid)
    sphere_y = np.sin(phi_grid) * np.sin(theta_grid)
    sphere_z = np.cos(phi_grid) * RJ_polar_ratio

    sphere_trace = go.Surface(
        x=sphere_x,
        y=sphere_y,
        z=sphere_z,
        surfacecolor=np.zeros_like(sphere_x),
        colorscale=[[0, "lightgray"], [1, "lightgray"]],
        showscale=False,
        opacity=1.0,
        hoverinfo="skip",
        name="Unit sphere",
        lighting=dict(
            ambient=0.55,
            diffuse=0.8,
            specular=0.3,
            roughness=0.7,
        ),
    )

    sc_trajectory = go.Scatter3d(
        x=A[:, 0],
        y=A[:, 1],
        z=A[:, 2],
        mode="lines",
        line=dict(color="black", width=2),
        name="Juno trajectory",
        hoverinfo="skip",
    )

    C_flat = C[D, :]
    points_trace = go.Scatter3d(
        x=C_flat[:, 0],
        y=C_flat[:, 1],
        z=C_flat[:, 2],
        mode="markers",
        marker=dict(
            size=1,
            color="darkorange",
            opacity=0.9,
            line=dict(color="black", width=0.5),
        ),
        name="All M-shell intersections",
        hoverinfo="skip",
    )

    def traces_for_time(t):
        """Create the dynamic traces for a particular time index."""
        position = A[t]
        endpoint = A[t] + B[t]

        # Select only the C points marked valid by row t of D.
        valid_points = C[t, D[t]]

        if valid_points.size == 0:
            cx, cy, cz = [], [], []
        else:
            # Ignore non-finite points even if the mask marks them valid.
            valid_points = valid_points[np.all(np.isfinite(valid_points), axis=1)]
            cx = valid_points[:, 0]
            cy = valid_points[:, 1]
            cz = valid_points[:, 2]

        position_trace = go.Scatter3d(
            x=[position[0]],
            y=[position[1]],
            z=[position[2]],
            mode="markers",
            marker=dict(
                size=7,
                color="crimson",
                line=dict(color="black", width=1),
            ),
            name="Juno",
            customdata=[[t]],
            hovertemplate=(
                "A<br>"
                "time=%{customdata[0]}<br>"
                "x=%{x:.4f}<br>"
                "y=%{y:.4f}<br>"
                "z=%{z:.4f}<extra></extra>"
            ),
        )

        # Exact line segment from A[t] to A[t] + B[t].
        vector_line_trace = go.Scatter3d(
            x=[position[0], endpoint[0]],
            y=[position[1], endpoint[1]],
            z=[position[2], endpoint[2]],
            mode="lines",
            line=dict(color="royalblue", width=8),
            name="Boresight",
            hoverinfo="skip",
        )

        # Arrowhead at A[t] + B[t].
        vector_tip_trace = go.Cone(
            x=[endpoint[0]],
            y=[endpoint[1]],
            z=[endpoint[2]],
            u=[B[t, 0]],
            v=[B[t, 1]],
            w=[B[t, 2]],
            anchor="tip",
            sizemode="absolute",
            sizeref=0.20,
            colorscale=[[0, "royalblue"], [1, "royalblue"]],
            showscale=False,
            name="B arrowhead",
            hovertemplate=(
                "A + B<br>"
                f"time={t}<br>"
                "x=%{x:.4f}<br>"
                "y=%{y:.4f}<br>"
                "z=%{z:.4f}<extra></extra>"
            ),
        )

        points_trace = go.Scatter3d(
            x=cx,
            y=cy,
            z=cz,
            mode="markers",
            marker=dict(
                size=4,
                color="darkorange",
                opacity=0.9,
                line=dict(color="black", width=0.5),
            ),
            name="M-shell intersections",
            customdata=np.full((len(cx), 1), t),
            hovertemplate=(
                "C point<br>"
                "time=%{customdata[0]}<br>"
                "x=%{x:.4f}<br>"
                "y=%{y:.4f}<br>"
                "z=%{z:.4f}<extra></extra>"
            ),
        )

        return [
            position_trace,
            vector_line_trace,
            vector_tip_trace,
            points_trace,
        ]

    # ------------------------------------------------------------------
    # Calculate fixed, equal axis ranges so the view does not rescale
    # as time changes.
    # ------------------------------------------------------------------
    coordinate_sets = [
        A.reshape(-1, 3),
        (A + B).reshape(-1, 3),
        np.array(
            [
                [-1.0, -1.0, -1.0],
                [1.0, 1.0, 1.0],
            ]
        ),
    ]

    masked_c = C[D]
    if masked_c.size:
        coordinate_sets.append(masked_c.reshape(-1, 3))

    all_points = np.vstack(coordinate_sets)
    all_points = all_points[np.all(np.isfinite(all_points), axis=1)]

    lower = np.min(all_points, axis=0)
    upper = np.max(all_points, axis=0)
    center = 0.5 * (lower + upper)

    # Use the same span on all axes.
    half_span = 0.5 * np.max(upper - lower)
    half_span = max(half_span, 1.0) * 1.1

    axis_ranges = [
        [center[i] - half_span, center[i] + half_span]
        for i in range(3)
    ]

    # Initial traces, including the static sphere.
    initial_dynamic_traces = traces_for_time(0)
    fig = go.Figure(
        data=initial_dynamic_traces + [sphere_trace, sc_trajectory, points_trace]
    )

    # Each frame updates only the first four dynamic traces.
    fig.frames = [
        go.Frame(
            name=str(t),
            data=traces_for_time(t),
            traces=[0, 1, 2, 3],
        )
        for t in range(N)
    ]

    slider_steps = [
        {
            "label": str(t),
            "method": "animate",
            "args": [
                [str(t)],
                {
                    "mode": "immediate",
                    "transition": {"duration": 0},
                    "frame": {"duration": 0, "redraw": True},
                },
            ],
        }
        for t in range(N)
    ]

    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
        ),
        template="plotly_white",
        showlegend=True,
        uirevision="preserve-camera",
        scene=dict(
            xaxis=dict(title="X", range=axis_ranges[0]),
            yaxis=dict(title="Y", range=axis_ranges[1]),
            zaxis=dict(title="Z", range=axis_ranges[2]),
            aspectmode="cube",
            camera=dict(
                eye=dict(x=1.6, y=1.6, z=1.2),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1),
                projection=dict(type="perspective"),
            ),
            dragmode="orbit",
        ),
        legend=dict(
            x=1.0,
            y=1.0,
            xanchor="right",
            yanchor="top",
            bgcolor="rgba(255, 255, 255, 0.85)",
            bordercolor="rgba(0, 0, 0, 0.25)",
            borderwidth=1,
        ),
        # Extra top margin provides room for the custom controls.
        margin=dict(l=0, r=0, b=100, t=125),
        sliders=[
            {
                "active": 0,
                "currentvalue": {
                    "prefix": "Time index: ",
                    "font": {"size": 16},
                },
                "pad": {"t": 55},
                "steps": slider_steps,
            }
        ],
        updatemenus=[],
    )

    wasd_camera_script = r"""
    (function () {
        const gd = document.getElementById('{plot_id}');

        const baseSpeed = 0.08;
        let camera = null;

        function clone(obj) {
            return JSON.parse(JSON.stringify(obj));
        }

        function add(a, b) {
            return {
                x: a.x + b.x,
                y: a.y + b.y,
                z: a.z + b.z
            };
        }

        function subtract(a, b) {
            return {
                x: a.x - b.x,
                y: a.y - b.y,
                z: a.z - b.z
            };
        }

        function multiply(v, scalar) {
            return {
                x: v.x * scalar,
                y: v.y * scalar,
                z: v.z * scalar
            };
        }

        function magnitude(v) {
            return Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
        }

        function normalize(v) {
            const length = magnitude(v);

            if (length < 1e-12) {
                return {x: 0, y: 0, z: 0};
            }

            return {
                x: v.x / length,
                y: v.y / length,
                z: v.z / length
            };
        }

        function cross(a, b) {
            return {
                x: a.y * b.z - a.z * b.y,
                y: a.z * b.x - a.x * b.z,
                z: a.x * b.y - a.y * b.x
            };
        }

        function getCamera() {
            if (camera !== null) {
                return clone(camera);
            }

            if (
                gd.layout.scene &&
                gd.layout.scene.camera
            ) {
                return clone(gd.layout.scene.camera);
            }

            return {
                eye: {x: 1.6, y: 1.6, z: 1.2},
                center: {x: 0, y: 0, z: 0},
                up: {x: 0, y: 0, z: 1}
            };
        }

        // Keep the keyboard camera synchronized with mouse-driven changes.
        gd.on('plotly_relayout', function (eventData) {
            if (eventData['scene.camera']) {
                camera = clone(eventData['scene.camera']);
            }
        });

        document.addEventListener('keydown', function (event) {
            // Do not capture keys while typing in an input field.
            const tag = document.activeElement.tagName.toLowerCase();

            if (
                tag === 'input' ||
                tag === 'textarea' ||
                tag === 'select'
            ) {
                return;
            }

            const key = event.key.toLowerCase();

            if (!['w', 'a', 's', 'd', 'q', 'e'].includes(key)) {
                return;
            }

            event.preventDefault();

            const cam = getCamera();

            cam.eye = cam.eye || {x: 1.6, y: 1.6, z: 1.2};
            cam.center = cam.center || {x: 0, y: 0, z: 0};
            cam.up = normalize(cam.up || {x: 0, y: 0, z: 1});

            // Direction in which the camera is looking.
            const forward = normalize(subtract(cam.center, cam.eye));

            // Camera-relative right direction.
            let right = normalize(cross(forward, cam.up));

            // Fallback if forward and up are nearly parallel.
            if (magnitude(right) < 1e-12) {
                right = {x: 1, y: 0, z: 0};
            }

            const speed = event.shiftKey ? baseSpeed * 3.0 : baseSpeed;

            let movement = {x: 0, y: 0, z: 0};

            switch (key) {
                case 'w':
                    movement = multiply(forward, speed);
                    break;
                case 's':
                    movement = multiply(forward, -speed);
                    break;
                case 'a':
                    movement = multiply(right, -speed);
                    break;
                case 'd':
                    movement = multiply(right, speed);
                    break;
                case 'q':
                    movement = multiply(cam.up, -speed);
                    break;
                case 'e':
                    movement = multiply(cam.up, speed);
                    break;
            }

            // Move both eye and center to translate the camera without
            // changing its viewing direction.
            cam.eye = add(cam.eye, movement);
            cam.center = add(cam.center, movement);

            camera = clone(cam);

            Plotly.relayout(gd, {
                'scene.camera': cam
            });
        });
    })();
    """

    playback_controls_script = r"""
    (function () {
        const gd = document.getElementById('{plot_id}');
        const baseFrameDuration = __BASE_FRAME_DURATION__;

        let playbackSpeed = 1.0;
        let isPlaying = false;

        // ------------------------------------------------------------
        // Construct the control panel.
        // ------------------------------------------------------------
        const panel = document.createElement('div');

        panel.style.position = 'absolute';
        panel.style.top = '58px';
        panel.style.left = '16px';
        panel.style.zIndex = '1000';

        panel.style.display = 'flex';
        panel.style.alignItems = 'center';
        panel.style.gap = '10px';

        panel.style.padding = '8px 12px';
        panel.style.background = 'rgba(255, 255, 255, 0.94)';
        panel.style.border = '1px solid #aaa';
        panel.style.borderRadius = '6px';
        panel.style.boxShadow = '0 2px 6px rgba(0, 0, 0, 0.18)';

        panel.style.fontFamily = 'Arial, sans-serif';
        panel.style.fontSize = '14px';

        // Keep the panel positioned relative to the Plotly container.
        const parent = gd.parentElement;
        const parentStyle = window.getComputedStyle(parent);

        if (parentStyle.position === 'static') {
            parent.style.position = 'relative';
        }

        // ------------------------------------------------------------
        // Play button.
        // ------------------------------------------------------------
        const playButton = document.createElement('button');
        playButton.type = 'button';
        playButton.textContent = '▶ Play';
        playButton.title = 'Play animation';

        // ------------------------------------------------------------
        // Pause button.
        // ------------------------------------------------------------
        const pauseButton = document.createElement('button');
        pauseButton.type = 'button';
        pauseButton.textContent = '■ Pause';
        pauseButton.title = 'Pause animation';

        // Apply common button styling.
        [playButton, pauseButton].forEach(function (button) {
            button.style.padding = '5px 10px';
            button.style.border = '1px solid #888';
            button.style.borderRadius = '4px';
            button.style.background = '#f7f7f7';
            button.style.color = '#222';
            button.style.fontSize = '14px';
            button.style.cursor = 'pointer';

            button.addEventListener('mouseenter', function () {
                button.style.background = '#e8e8e8';
            });

            button.addEventListener('mouseleave', function () {
                button.style.background = '#f7f7f7';
            });
        });

        // ------------------------------------------------------------
        // Vertical separator.
        // ------------------------------------------------------------
        const separator = document.createElement('div');
        separator.style.width = '1px';
        separator.style.height = '28px';
        separator.style.background = '#bbb';

        // ------------------------------------------------------------
        // Speed label.
        // ------------------------------------------------------------
        const speedLabel = document.createElement('label');
        speedLabel.htmlFor = 'plotly-speed-slider-{plot_id}';
        speedLabel.textContent = 'Speed:';
        speedLabel.style.whiteSpace = 'nowrap';

        // ------------------------------------------------------------
        // Speed slider.
        // ------------------------------------------------------------
        const speedSlider = document.createElement('input');
        speedSlider.id = 'plotly-speed-slider-{plot_id}';
        speedSlider.type = 'range';
        speedSlider.min = '0.25';
        speedSlider.max = '4.0';
        speedSlider.step = '0.25';
        speedSlider.value = '1.0';
        speedSlider.style.width = '160px';
        speedSlider.title = 'Playback speed';

        // ------------------------------------------------------------
        // Numerical speed display.
        // ------------------------------------------------------------
        const speedDisplay = document.createElement('span');
        speedDisplay.textContent = '1.00×';
        speedDisplay.style.minWidth = '46px';
        speedDisplay.style.fontWeight = 'bold';
        speedDisplay.style.textAlign = 'right';

        panel.appendChild(playButton);
        panel.appendChild(pauseButton);
        panel.appendChild(separator);
        panel.appendChild(speedLabel);
        panel.appendChild(speedSlider);
        panel.appendChild(speedDisplay);

        parent.appendChild(panel);

        // ------------------------------------------------------------
        // Update speed.
        // ------------------------------------------------------------
        speedSlider.addEventListener('input', function () {
            playbackSpeed = Number(speedSlider.value);
            speedDisplay.textContent = playbackSpeed.toFixed(2) + '×';
        });

        // ------------------------------------------------------------
        // Start or resume playback.
        // ------------------------------------------------------------
        playButton.addEventListener('click', function () {
            if (isPlaying) {
                return;
            }

            isPlaying = true;

            const frameDuration = Math.max(
                1,
                Math.round(baseFrameDuration / playbackSpeed)
            );

            Plotly.animate(
                gd,
                null,
                {
                    fromcurrent: true,
                    mode: 'immediate',
                    transition: {
                        duration: 0
                    },
                    frame: {
                        duration: frameDuration,
                        redraw: true
                    }
                }
            ).finally(function () {
                isPlaying = false;
            });
        });

        // ------------------------------------------------------------
        // Pause playback.
        // ------------------------------------------------------------
        pauseButton.addEventListener('click', function () {
            isPlaying = false;

            Plotly.animate(
                gd,
                [null],
                {
                    mode: 'immediate',
                    transition: {
                        duration: 0
                    },
                    frame: {
                        duration: 0,
                        redraw: false
                    }
                }
            );
        });

        // Update the internal state when the animation reaches its end.
        gd.on('plotly_animated', function () {
            isPlaying = false;
        });
    })();
    """

    if output_html is not None:
        playback_script_for_figure = playback_controls_script.replace(
            "__BASE_FRAME_DURATION__",
            str(frame_duration_ms),
        )

        fig.write_html(
            output_html,
            include_plotlyjs=True,
            auto_open=False,
            post_script=[
                wasd_camera_script,
                playback_script_for_figure,
            ],
        )

    return fig


def main():
    N = 100
    M = 30

    t = np.linspace(0, 4 * np.pi, N)

    A = np.column_stack((
        2.0 * np.cos(t),
        2.0 * np.sin(t),
        0.1 * t,
    ))

    # Example unit vectors.
    B = np.column_stack((
        -np.sin(t),
        np.cos(t),
        np.full(N, 0.5),
    ))
    B /= np.linalg.norm(B, axis=1, keepdims=True)

    # Example point clouds and masks.
    C = A[:, None, :] + 0.5 * np.random.randn(N, M, 3)
    D = np.random.random((N, M)) > 0.25

    fig = create_3d_time_viewer(
        A,
        B,
        C,
        D,
        title="Interactive 3D Time-Series Viewer",
        frame_duration_ms=75,
        output_html="viewer.html",  # Optional
    )

    fig.show()


if __name__ == "__main__":
    main()
