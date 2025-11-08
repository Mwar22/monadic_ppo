from mujoco import MjModel
from etils import epath

xml_path = epath.Path("/home/lucas/Documentos/MLProjects/monadic_ppo/model/thor_robot_collision_only.xml")
xml_text = xml_path.read_text()

mj_model = MjModel.from_xml_string(xml_text)  # no assets
print("Model loaded OK:", mj_model.nbody)