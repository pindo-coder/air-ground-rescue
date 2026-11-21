"""Build the D scene's redesigned Webots 4WD rover & quadrotor drone and their Gazebo devices."""

from itertools import chain
from webots_static_scene import Parameter, Placement, element, number_list


def _resolve(value, parameters):
    if isinstance(value, Parameter):
        return parameters[value.name]
    return value


def _camera_sensor_name(source_name, is_depth):
    if is_depth:
        return "depth"
    if source_name in {"rgb", "aerial_rgb"}:
        return "rgb"
    if source_name == "rover_chase_cam":
        return "chase_cam"
    if source_name == "drone_fpv_cam":
        return "fpv_cam"
    return source_name


def add_camera(link, source, namespace, rate):
    fields = source.fields
    depth = source.kind == "RangeFinder"
    src_name = fields.get("name", "depth" if depth else "rgb")
    name = _camera_sensor_name(src_name, depth)
    kind = "depth_camera" if depth else "camera"
    effective_rate = 5 if name in {"chase_cam", "fpv_cam"} else rate
    sensor = element(link, "sensor", name=name, type=kind)
    Placement().child(source).write(sensor)
    element(sensor, "topic", f"/{namespace}/{name}")
    element(sensor, "gz_frame_id", f"{namespace}/{name}")
    element(sensor, "always_on", "true")
    element(sensor, "update_rate", str(effective_rate))
    camera = element(sensor, "camera")
    element(camera, "horizontal_fov", str(fields["fieldOfView"]))
    image = element(camera, "image")
    for field in ("width", "height"):
        element(image, field, str(int(fields[field])))
    if not depth:
        element(image, "format", "R8G8B8")
        noise_std = float(fields.get("noise", 0.0))
        if noise_std > 0.0:
            noise = element(camera, "noise")
            element(noise, "type", "gaussian")
            element(noise, "stddev", str(noise_std))
    clip = element(camera, "clip")
    element(clip, "near", str(fields["minRange" if depth else "near"]))
    element(clip, "far", str(fields["maxRange" if depth else "far"]))


def add_lidar(link, source):
    fields = source.fields
    sensor = element(link, "sensor", name="lidar", type="gpu_lidar")
    Placement().child(source).write(sensor)
    element(sensor, "topic", "/rover/lidar")
    element(sensor, "gz_frame_id", "rover/lidar")
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
    noise_std = float(fields.get("noise", 0.0))
    if noise_std > 0.0:
        noise = element(lidar, "noise")
        element(noise, "type", "gaussian")
        element(noise, "stddev", str(noise_std))


def add_imu(link, source, namespace, rate=10):
    fields = source.fields
    sensor = element(link, "sensor", name="attitude", type="imu")
    Placement().child(source).write(sensor)
    element(sensor, "topic", f"/{namespace}/imu")
    element(sensor, "gz_frame_id", f"{namespace}/imu")
    element(sensor, "always_on", "true")
    element(sensor, "update_rate", str(rate))
    imu = element(sensor, "imu")
    ref = element(imu, "orientation_reference_frame")
    element(ref, "localization", "ENU")
    noise_std = float(fields.get("noise", 0.0))
    if noise_std > 0.0:
        ang = element(imu, "angular_velocity")
        for axis in ("x", "y", "z"):
            n = element(element(ang, axis), "noise", type="gaussian")
            element(n, "mean", "0.0")
            element(n, "stddev", str(noise_std))


def add_gps(link, source, namespace, rate=10):
    """Export NavSat sensor at exact Webots GPS mount pose.

    Note: Horizontal noise is intentionally omitted from SDF to avoid gz-sensors
    degree-vs-meter unit ambiguity (Issue #325); exact 0.04 m ENU noise is
    applied deterministically in the ROS 2 NavSatToEnu adapter.
    """
    sensor = element(link, "sensor", name="gps", type="navsat")
    Placement().child(source).write(sensor)
    element(sensor, "topic", f"/{namespace}/gps")
    element(sensor, "gz_frame_id", f"{namespace}/gps")
    element(sensor, "always_on", "true")
    element(sensor, "update_rate", str(rate))
    element(sensor, "navsat")


def add_contact_sensor(link, collision_names, namespace="rover", rate=10):
    if not collision_names:
        return
    sensor = element(link, "sensor", name="chassis_contact", type="contact")
    element(sensor, "topic", f"/{namespace}/contact")
    element(sensor, "always_on", "true")
    element(sensor, "update_rate", str(rate))
    contact = element(sensor, "contact")
    for cname in collision_names:
        element(contact, "collision", cname)


def add_inertia(link, mass, moments, center=(0, 0, 0)):
    body = element(link, "inertial")
    element(body, "mass", str(mass))
    element(body, "pose", number_list((*center, 0, 0, 0)))
    inertia = element(body, "inertia")
    for axis, moment in zip(("ixx", "iyy", "izz"), moments):
        element(inertia, axis, str(moment))


def write_body(exporter, body, link, parameters=None):
    visuals = chain.from_iterable(
        exporter.shapes(node, Placement(), parameters=parameters)
        for node in body.fields.get("children", [])
        if node.kind in {"Pose", "Shape", "Group", "RescueRoverBody"}
    )
    collisions = exporter.shapes(body.fields.get("boundingObject"), Placement(), collision=True, parameters=parameters)
    exporter.write_parts(link, visuals, collisions)


def write_friction(link, coefficient):
    for collision in link.findall("collision"):
        ode = element(element(element(collision, "surface"), "friction"), "ode")
        element(ode, "mu", str(coefficient))
        element(ode, "mu2", str(coefficient))


def _wheel_link_name(motor_name):
    if "::" in motor_name:
        side_part, pos_part = motor_name.split("::", 1)
        side = side_part.split("_", 1)[0]
        return f"{pos_part}_{side}_wheel"
    side = motor_name.split("_", 1)[0]
    return f"{side}_wheel"


def add_wheel(exporter, model, source, friction, parameters=None):
    body = source.fields["endPoint"]
    motor = next(device for device in source.fields["device"] if device.kind == "RotationalMotor")
    motor_name = _resolve(motor.fields["name"], parameters)
    name = _wheel_link_name(motor_name)
    link = element(model, "link", name=name)
    Placement().child(body, parameters).write(link)
    write_body(exporter, body, link, parameters=parameters)
    write_friction(link, friction)
    mass = body.fields["physics"].fields["mass"]
    cylinder = body.fields["boundingObject"].fields
    radius, length = cylinder["radius"], cylinder["height"]
    transverse = mass * (3 * radius**2 + length**2) / 12
    add_inertia(link, mass, (transverse, transverse, 0.5 * mass * radius**2))

    joint = element(model, "joint", name=f"{name}_joint", type="revolute")
    params = source.fields["jointParameters"].fields
    anchor = _resolve(params["anchor"], parameters)
    element(joint, "pose", number_list((*anchor, 0, 0, 0)), relative_to="__model__")
    element(joint, "parent", "chassis")
    element(joint, "child", name)
    axis = element(joint, "axis")
    element(axis, "xyz", number_list(params["axis"]), expressed_in="__model__")
    limits = element(axis, "limit")
    element(limits, "effort", str(motor.fields["maxTorque"]))
    element(limits, "velocity", str(motor.fields["maxVelocity"]))
    element(element(axis, "dynamics"), "damping", str(params["dampingConstant"]))
    return joint.attrib["name"], anchor, radius


def add_rover(world, source, exporter, friction):
    body, fields = exporter.proto_instance(source)
    model = element(world, "model", name="rover")
    Placement().child(body, fields).write(model)
    chassis = element(model, "link", name="chassis")
    write_body(exporter, body, chassis, parameters=fields)
    contact_mat = body.fields.get("contactMaterial", "default")
    if contact_mat in friction:
        write_friction(chassis, friction[contact_mat])
    physics = body.fields["physics"].fields
    # Approximate the Webots auto-inertia with the chassis box about the specified COM.
    mass, center = physics["mass"], physics["centerOfMass"]
    box = body.fields["boundingObject"].fields["children"][0]
    x, y, z = box.fields["children"][0].fields["size"] if box.kind == "Pose" else box.fields["size"]
    box_z = box.fields["translation"][2] if box.kind == "Pose" else body.fields["boundingObject"].fields.get("translation", (0, 0, 0))[2]
    offset = box_z - center[2]
    add_inertia(chassis, mass, (mass * ((y*y + z*z) / 12 + offset**2),
                               mass * ((x*x + z*z) / 12 + offset**2),
                               mass * (x*x + y*y) / 12), center)

    chassis_collisions = [c.attrib["name"] for c in chassis.findall("collision")]
    add_contact_sensor(chassis, chassis_collisions, namespace="rover", rate=10)

    wheels = {}
    left_joints = []
    right_joints = []
    for child in body.fields["children"]:
        if child.kind in {"Camera", "RangeFinder"}:
            add_camera(chassis, child, "rover", rate=10)
        elif child.kind == "Lidar":
            add_lidar(chassis, child)
        elif child.kind == "InertialUnit":
            add_imu(chassis, child, "rover", rate=10)
        elif child.kind == "GPS":
            add_gps(chassis, child, "rover", rate=10)
        elif child.kind == "HingeJoint":
            joint, anchor, radius = add_wheel(exporter, model, child, friction.get("rubber", 0.95))
            wheels[joint] = (anchor, radius)
            (left_joints if anchor[1] > 0 else right_joints).append(joint)
        elif child.kind == "RescueRoverWheel":
            wheel_body, wheel_fields = exporter.proto_instance(child)
            joint, anchor, radius = add_wheel(
                exporter, model, wheel_body, friction.get("rubber", 0.95), parameters=wheel_fields
            )
            wheels[joint] = (anchor, radius)
            (left_joints if anchor[1] > 0 else right_joints).append(joint)

    if not left_joints or not right_joints:
        raise ValueError("ResearchRover export requires at least one left and one right drive wheel.")

    left_y = sum(wheels[j][0][1] for j in left_joints) / len(left_joints)
    right_y = sum(wheels[j][0][1] for j in right_joints) / len(right_joints)
    wheel_radius = wheels[left_joints[0]][1]

    drive = element(model, "plugin", filename="gz-sim-diff-drive-system",
                    name="gz::sim::systems::DiffDrive")
    for j in left_joints:
        element(drive, "left_joint", j)
    for j in right_joints:
        element(drive, "right_joint", j)
    element(drive, "wheel_separation", f"{left_y - right_y:.6g}")
    element(drive, "wheel_radius", f"{wheel_radius:.6g}")
    element(drive, "max_linear_velocity", f"{wheel_radius * 10:.6g}")
    element(drive, "topic", "/rover/cmd_vel")
    element(drive, "odom_topic", "/rover/odom")
    element(drive, "tf_topic", "/model/rover/tf")
    element(drive, "frame_id", "odom")
    element(drive, "child_frame_id", "base_link")

    joints = element(model, "plugin", filename="gz-sim-joint-state-publisher-system",
                     name="gz::sim::systems::JointStatePublisher")
    element(joints, "topic", "/rover/joint_states")
    for joint in (*left_joints, *right_joints):
        element(joints, "joint_name", joint)


def add_drone(world, source, exporter, image_period):
    body, fields = exporter.proto_instance(source)
    model = element(world, "model", name="drone")
    Placement().child(body, fields).write(model)
    element(model, "static", "true")
    link = element(model, "link", name="body")
    write_body(exporter, body, link, parameters=fields)
    for child in body.fields["children"]:
        if child.kind in {"Camera", "RangeFinder"}:
            add_camera(link, child, "drone", rate=1 / image_period)
        elif child.kind == "GPS":
            add_gps(link, child, "drone", rate=10)
        elif child.kind == "InertialUnit":
            add_imu(link, child, "drone", rate=10)
