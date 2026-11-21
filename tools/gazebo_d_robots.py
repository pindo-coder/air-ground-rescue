"""Build the D scene's two Webots platforms and their Gazebo devices."""

from itertools import chain
from webots_static_scene import Placement, element, number_list


def add_camera(link, source, namespace, rate):
    fields = source.fields
    depth = source.kind == "RangeFinder"
    name, kind = ("depth", "depth_camera") if depth else ("rgb", "camera")
    sensor = element(link, "sensor", name=name, type=kind)
    Placement().child(source).write(sensor)
    element(sensor, "topic", f"/{namespace}/{name}")
    element(sensor, "always_on", "true")
    element(sensor, "update_rate", str(rate))
    camera = element(sensor, "camera")
    element(camera, "horizontal_fov", str(fields["fieldOfView"]))
    image = element(camera, "image")
    for field in ("width", "height"):
        element(image, field, str(int(fields[field])))
    if not depth:
        element(image, "format", "R8G8B8")
        noise = element(camera, "noise")
        element(noise, "type", "gaussian")
        element(noise, "stddev", str(fields["noise"]))
    clip = element(camera, "clip")
    element(clip, "near", str(fields["minRange" if depth else "near"]))
    element(clip, "far", str(fields["maxRange" if depth else "far"]))


def add_lidar(link, source):
    fields = source.fields
    sensor = element(link, "sensor", name="lidar", type="gpu_lidar")
    Placement().child(source).write(sensor)
    element(sensor, "topic", "/rover/lidar")
    element(sensor, "always_on", "true")
    element(sensor, "update_rate", "10")
    lidar = element(sensor, "lidar")
    scan = element(lidar, "scan")
    for name, samples, fov in (
        ("horizontal", fields["horizontalResolution"], fields["fieldOfView"]),
        ("vertical", fields["numberOfLayers"], fields["verticalFieldOfView"]),
    ):
        axis = element(scan, name)
        element(axis, "samples", str(int(samples)))
        element(axis, "min_angle", str(-fov / 2))
        element(axis, "max_angle", str(fov / 2))
    distance = element(lidar, "range")
    element(distance, "min", str(fields["minRange"]))
    element(distance, "max", str(fields["maxRange"]))
    element(distance, "resolution", "0.001")
    noise = element(lidar, "noise")
    element(noise, "type", "gaussian")
    element(noise, "stddev", str(fields["noise"]))


def add_inertia(link, mass, moments, center=(0, 0, 0)):
    body = element(link, "inertial")
    element(body, "mass", str(mass))
    element(body, "pose", number_list((*center, 0, 0, 0)))
    inertia = element(body, "inertia")
    for axis, moment in zip(("ixx", "iyy", "izz"), moments):
        element(inertia, axis, str(moment))


def write_body(exporter, body, link):
    visuals = chain.from_iterable(
        exporter.shapes(node, Placement())
        for node in body.fields["children"] if node.kind in {"Pose", "Shape"}
    )
    collisions = exporter.shapes(body.fields["boundingObject"], Placement(), collision=True)
    exporter.write_parts(link, visuals, collisions)


def write_friction(link, coefficient):
    for collision in link.findall("collision"):
        ode = element(element(element(collision, "surface"), "friction"), "ode")
        element(ode, "mu", str(coefficient))
        element(ode, "mu2", str(coefficient))


def add_wheel(exporter, model, source, friction):
    body = source.fields["endPoint"]
    motor = next(device for device in source.fields["device"] if device.kind == "RotationalMotor")
    side = motor.fields["name"].split("_", 1)[0]
    name = f"{side}_wheel"
    link = element(model, "link", name=name)
    Placement().child(body).write(link)
    write_body(exporter, body, link)
    write_friction(link, friction)
    mass = body.fields["physics"].fields["mass"]
    cylinder = body.fields["boundingObject"].fields
    radius, length = cylinder["radius"], cylinder["height"]
    transverse = mass * (3 * radius**2 + length**2) / 12
    add_inertia(link, mass, (transverse, transverse, 0.5 * mass * radius**2))

    joint = element(model, "joint", name=f"{name}_joint", type="revolute")
    params = source.fields["jointParameters"].fields
    element(joint, "pose", number_list((*params["anchor"], 0, 0, 0)), relative_to="__model__")
    element(joint, "parent", "chassis")
    element(joint, "child", name)
    axis = element(joint, "axis")
    element(axis, "xyz", number_list(params["axis"]), expressed_in="__model__")
    limits = element(axis, "limit")
    element(limits, "effort", str(motor.fields["maxTorque"]))
    element(limits, "velocity", str(motor.fields["maxVelocity"]))
    element(element(axis, "dynamics"), "damping", str(params["dampingConstant"]))
    return joint.attrib["name"], params["anchor"][1], radius


def add_rover(world, source, exporter, friction):
    body, fields = exporter.proto_instance(source)
    model = element(world, "model", name="rover")
    Placement().child(body, fields).write(model)
    chassis = element(model, "link", name="chassis")
    write_body(exporter, body, chassis)
    write_friction(chassis, friction[body.fields["contactMaterial"]])
    physics = body.fields["physics"].fields
    # Approximate the Webots auto-inertia with the chassis box about the specified COM.
    mass, center = physics["mass"], physics["centerOfMass"]
    box = body.fields["boundingObject"].fields["children"][0]
    x, y, z = box.fields["children"][0].fields["size"]
    offset = box.fields["translation"][2] - center[2]
    add_inertia(chassis, mass, (mass * ((y*y + z*z) / 12 + offset**2),
                               mass * ((x*x + z*z) / 12 + offset**2),
                               mass * (x*x + y*y) / 12), center)
    wheels = {}
    for child in body.fields["children"]:
        if child.kind in {"Camera", "RangeFinder"}:
            add_camera(chassis, child, "rover", rate=10)
        elif child.kind == "Lidar":
            add_lidar(chassis, child)
        elif child.kind == "InertialUnit":
            sensor = element(chassis, "sensor", name="attitude", type="imu")
            element(sensor, "topic", "/rover/imu")
            element(sensor, "always_on", "true")
            element(sensor, "update_rate", "10")
        elif child.kind == "HingeJoint":
            joint, anchor, radius = add_wheel(exporter, model, child, friction["rubber"])
            wheels[joint] = (anchor, radius)

    left, right = "left_wheel_joint", "right_wheel_joint"
    drive = element(model, "plugin", filename="gz-sim-diff-drive-system",
                    name="gz::sim::systems::DiffDrive")
    element(drive, "left_joint", left)
    element(drive, "right_joint", right)
    element(drive, "wheel_separation", str(wheels[left][0] - wheels[right][0]))
    element(drive, "wheel_radius", str(wheels[left][1]))
    element(drive, "max_linear_velocity", str(wheels[left][1] * 10))
    element(drive, "topic", "/rover/cmd_vel")
    element(drive, "odom_topic", "/rover/odom")
    joints = element(model, "plugin", filename="gz-sim-joint-state-publisher-system",
                     name="gz::sim::systems::JointStatePublisher")
    element(joints, "topic", "/rover/joint_states")
    for joint in (left, right):
        element(joints, "joint_name", joint)


def add_drone(world, source, exporter, image_period):
    body, fields = exporter.proto_instance(source)
    model = element(world, "model", name="drone")
    Placement().child(body, fields).write(model)
    element(model, "static", "true")
    link = element(model, "link", name="body")
    write_body(exporter, body, link)
    for child in body.fields["children"]:
        if child.kind in {"Camera", "RangeFinder"}:
            add_camera(link, child, "drone", rate=1 / image_period)
