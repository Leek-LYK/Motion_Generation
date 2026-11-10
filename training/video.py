"""Headless MuJoCo kinematic replay with provenance and root-trajectory HUD."""
import json
import subprocess
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def render(qpos, output, metadata, xml_path, fps=30):
    import mujoco
    output = Path(output)
    xml_path = Path(xml_path)
    xml = ET.parse(xml_path).getroot()
    compiler = xml.find("compiler")
    compiler.set("meshdir", str((xml_path.parent / compiler.get("meshdir")).resolve()))
    visual = xml.find("visual")
    if visual is None:
        visual = ET.SubElement(xml, "visual")
    global_vis = visual.find("global")
    if global_vis is None:
        global_vis = ET.SubElement(visual, "global")
    global_vis.set("offwidth", "960")
    global_vis.set("offheight", "540")
    asset = xml.find("asset")
    ET.SubElement(asset, "texture", name="m3_ground_texture", type="2d", builtin="checker",
                  rgb1="0.17 0.21 0.27", rgb2="0.23 0.28 0.35", width="512", height="512")
    ET.SubElement(asset, "material", name="m3_ground_material", texture="m3_ground_texture",
                  texrepeat="1 1", texuniform="true", reflectance="0.08")
    world = xml.find("worldbody")
    planes = [geom for body in xml.findall("worldbody") for geom in body.findall("geom")
              if geom.get("type") == "plane"]
    if planes:
        # The upstream MJCF has multiple worldbody blocks, including its own floor.
        for plane in planes:
            plane.set("material", "m3_ground_material")
    else:
        ET.SubElement(world, "geom", name="m3_ground", type="plane", size="0 0 0.1",
                      material="m3_ground_material", contype="0", conaffinity="0")
    ET.SubElement(world, "light", name="m3_key", pos="0 -3 5", dir="0 0 -1",
                  directional="true", diffuse="0.8 0.8 0.8", ambient="0.3 0.3 0.3")
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
    if qpos.ndim != 2 or qpos.shape[1] != model.nq or not np.isfinite(qpos).all():
        raise ValueError("Invalid qpos for replay")
    if np.max(np.abs(np.linalg.norm(qpos[:, 3:7], axis=1) - 1)) > 1e-4:
        raise ValueError("Non-unit root quaternion")
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.azimuth, camera.elevation, camera.distance = 135, -20, 2.7
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0  # Hide collision primitives, keep actual robot mesh.
    renderer = mujoco.Renderer(model, height=540, width=960)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 18)
    small = ImageFont.truetype(font_path, 15)
    title_font = ImageFont.truetype(font_path, 24)
    xy = qpos[:, :2]
    center = (xy.min(0) + xy.max(0)) / 2
    extent = max(float(np.ptp(xy, axis=0).max()), 1.0) * 1.2
    def plot_point(p):
        return (1115 + float(p[0] - center[0]) / extent * 260,
                315 - float(p[1] - center[1]) / extent * 260)
    trajectory = [plot_point(p) for p in xy]
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", "1280x720", "-r", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
               "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)]
    posters = []
    with output.with_suffix(".ffmpeg.log").open("wb") as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
        try:
            for index, q in enumerate(qpos):
                data.qpos[:] = q
                mujoco.mj_forward(model, data)  # Deliberately no mj_step / controller.
                camera.lookat[:] = [q[0], q[1], 0.65]
                renderer.update_scene(data, camera=camera, scene_option=option)
                for trail in qpos[:index + 1:3]:
                    if renderer.scene.ngeom >= renderer.scene.maxgeom:
                        break
                    geom = renderer.scene.geoms[renderer.scene.ngeom]
                    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([.012] * 3),
                                       np.array([trail[0], trail[1], .018]), np.eye(3).ravel(),
                                       np.array([1., .55, .12, 1.]))
                    renderer.scene.ngeom += 1
                frame = Image.new("RGB", (1280, 720), (16, 23, 34))
                frame.paste(Image.fromarray(renderer.render()), (0, 125))
                draw = ImageDraw.Draw(frame)
                draw.text((20, 12), "KIMODO G1  |  TEXT TO MOTION", font=title_font, fill=(235, 243, 255))
                for line_number, line in enumerate(textwrap.wrap(metadata["prompt"], 115)):
                    draw.text((20, 48 + 24 * line_number), line, font=font, fill=(235, 243, 255))
                draw.text((20, 100), f"Train {metadata['training_run_id']}  |  step {metadata['checkpoint_step']} "
                          f"{metadata['weights'].upper()}  |  seed {metadata['seed']}  |  CFG {metadata.get('guidance', 1):g}  |  {fps} FPS", font=small, fill=(160, 190, 220))
                draw.text((980, 142), "ROOT TRAJECTORY (XY)", font=font, fill="white")
                draw.rectangle((977, 177, 1253, 453), outline=(85, 105, 130))
                if len(trajectory) > 1:
                    draw.line(trajectory, fill=(80, 95, 115), width=2)
                    draw.line(trajectory[:index + 1] or trajectory[:1], fill=(255, 165, 45), width=3)
                sx, sy = trajectory[0]
                draw.ellipse((sx - 4, sy - 4, sx + 4, sy + 4), fill=(80, 200, 255))
                px, py = trajectory[index]
                draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=(255, 185, 60))
                draw.text((980, 465), f"Plot span: {extent:.2f} m; +X right, +Y up", font=small, fill=(185, 200, 220))
                draw.text((980, 495), f"Time  {index / fps:.2f} / {len(qpos) / fps:.2f} s", font=font, fill="white")
                draw.text((980, 525), f"Root height  {q[2]:.3f} m", font=font, fill="white")
                draw.text((980, 555), f"Travel  {np.linalg.norm(q[:2] - qpos[0, :2]):.3f} m", font=font, fill="white")
                processing = metadata.get('postprocess_label', 'no postprocess')
                draw.text((20, 682), f"KINEMATIC REPLAY: qpos + mj_forward | no controller / physics stepping | {processing}",
                          font=font, fill=(255, 193, 100))
                process.stdin.write(np.asarray(frame).tobytes())
                if index in {0, len(qpos)//2, len(qpos)-1}:
                    posters.append(frame.copy())
                if index == len(qpos)//2:
                    frame.save(output.with_name("poster.png"))
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError(f"ffmpeg failed; see {output.with_suffix('.ffmpeg.log')}")
        finally:
            renderer.close()
            if process.poll() is None:
                process.kill()
                process.wait()
    sheet = Image.new("RGB", (1280, 720 * len(posters)))
    for i, image in enumerate(posters):
        sheet.paste(image, (0, i * 720))
    sheet.save(output.with_name("contact_sheet.png"))
    probe = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                    "stream=width,height,nb_frames,avg_frame_rate,duration", "-of", "json", str(output)], text=True)
    stream = json.loads(probe)["streams"][0]
    if int(stream["nb_frames"]) != len(qpos):
        raise ValueError("Encoded frame count mismatch")
    return {"stream": stream, "renderer": "mujoco.Renderer/EGL", "replay": "kinematic mj_forward only",
            "camera": "follow root; azimuth 135; elevation -20; distance 2.7m", "physics_stepping": False}
