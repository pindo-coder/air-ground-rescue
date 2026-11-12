"""Collapsed masonry houses and flood-carried structural debris."""
import math


def add_flood_ruins(elements, make_part, make_link):
    shift = {"A": -.5, "B": .0, "C": 1.2}[elements.p["key"]]
    buildings = [
        ("HOUSE_BANK", (-7.5 + shift, -1.2), .27, 0),
        ("HOUSE_COLLAPSED", (1.9 + shift, -8.4), -.44, 1),
        ("HOUSE_FOUNDATION", (6.8 + shift, -.2), .64, 2),
    ]
    for name, xy, yaw, ruin in buildings:
        parts = []
        def wall(label, width, depth, height, at, rotation=(0, 0, 1, 0)):
            item = make_part(label, "JaggedWall", [width, depth, height], at,
                             ".78 .75 .65", rotation)
            item["material"] = "brick" if label in ("LEFT_BROKEN", "FRONT_RIGHT", "RIGHT_END") else "plaster"
            parts.append(item)

        # Door and window gaps, exposed wall edges and several broken wall heights.
        wall("BACK_L", 2.10, .22, 2.30 - .30 * ruin, [-1.18, 1.65, 1.15 - .15 * ruin])
        wall("BACK_R", 2.05, .22, 1.67 - .35 * ruin, [1.05, 1.65, .835 - .175 * ruin])
        wall("LEFT_PIER", .24, 1.05, 2.08, [-2.25, 1.0, 1.04])
        wall("LEFT_SILL", .24, 1.0, .72, [-2.25, -.06, .36])
        wall("LEFT_BROKEN", .24, .86, 1.50 - .30 * ruin, [-2.25, -1.11, .75 - .15 * ruin])
        wall("FRONT_LEFT", 1.30, .22, 1.73 - .32 * ruin, [-1.58, -1.65, .865 - .16 * ruin])
        wall("FRONT_RIGHT", 1.13, .22, .80, [.96, -1.65, .40])
        wall("RIGHT_END", .22, 1.0, 1.05, [2.25, 1.1, .525])
        slab = make_part("BROKEN_SLAB", "Box", [2.30, 1.25, .13], [.67, -.40, .11],
                         ".66 .63 .57", [.26, 1, .1, .11])
        slab["material"] = "plaster"
        parts.append(slab)

        for j, (a, b) in enumerate([
            ((-2.0, 1.45, 2.12), (1.22, .22, .60)),
            ((-1.75, -1.1, .14), (1.43, 1.25, 1.63)),
            ((.2, -1.5, .13), (1.9, .75, .35)),
        ]):
            beam = make_link(f"RAFTER_{j}", a, b, .065, ".40 .29 .17")
            beam["material"] = "wood"
            parts.append(beam)

        # Two roof fragments lying at different angles make the collapse legible.
        for j, (at, rot, dims) in enumerate([
            ([-.42, .32, 1.24 - .17 * ruin], [0, 1, .27, .39 + .12 * ruin], [3.70, 1.54, .055]),
            ([1.45, -.68, .46], [1, .24, .08, -.43], [2.45, 1.65, .055]),
        ]):
            panel = make_part(f"ROOF_{j}", "RoofPanel", dims, at,
                              ".38 .24 .17" if ruin != 2 else ".28 .29 .28", rot)
            panel["material"] = "roof"
            parts.append(panel)

        elements.add(name, "collapsed_house", xy, yaw, parts, [5.4, 4.2, 2.5], "detour")

        c, s = math.cos(yaw), math.sin(yaw)
        # Individual hard fragments are separately registered for collision metrics.
        for j in range(17):
            lx, ly = elements.rng.uniform(-3.2, 3.4), elements.rng.uniform(-3.25, 2.35)
            px, py = xy[0] + c * lx - s * ly, xy[1] + s * lx + c * ly
            width = elements.rng.uniform(.17, .60)
            depth = elements.rng.uniform(.15, .38)
            height = elements.rng.uniform(.10, .34)
            fragment = make_part("FRAGMENT", "BrokenBrick", [width, depth, height],
                                 [0, 0, height / 2], elements.rng.choice((".55 .34 .23", ".50 .47 .41", ".67 .57 .43")))
            fragment["material"] = "rock"
            elements.add(f"{name}_DEBRIS_{j}", "building_debris", (px, py),
                         elements.rng.uniform(-3.1, 3.1), [fragment],
                         [width, depth, height], "detour")
