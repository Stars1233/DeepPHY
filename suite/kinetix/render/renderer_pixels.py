import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from jax2d.engine import select_shape
from jax2d.maths import rmat
from jaxgl.maths import dist_from_line
from jaxgl.renderer import clear_screen, make_renderer
from jaxgl.shaders import (
    nearest_neighbour,
    fragment_shader_edged_quad,
    make_fragment_shader_convex_ngon_with_edges,
    fragment_shader_circle,
)

from suite.kinetix.environment.env_state import EnvState, StaticEnvParams
from suite.kinetix.render.textures import FJOINT_TEXTURE_6_RGBA, RJOINT_TEXTURE_6_RGBA, THRUSTER_TEXTURE_16_RGBA


def make_render_pixels(env_params, static_params: StaticEnvParams):
    screen_dim = static_params.screen_dim
    downscale = static_params.downscale

    joint_tex_size = 6
    thruster_tex_size = 16

    edge_thickness = 2
    use_textures_for_joints = True
    # if we downscale, the edges are too thick, and the textures do not show.
    if downscale == 2:
        edge_thickness = 1
    elif downscale == 4:
        edge_thickness = 0
        use_textures_for_joints = False
    fragment_shader_quad_no_edges = make_fragment_shader_convex_ngon_with_edges(4, edge_thickness=edge_thickness)

    FIXATED_COLOUR = jnp.array([80, 80, 80])
    JOINT_COLOURS = jnp.array(
        [
            # [0, 0, 255],
            [255, 255, 255],  # yellow
            [255, 255, 0],  # yellow
            [255, 0, 255],  # purple/magenta
            [0, 255, 255],  # cyan
            [255, 153, 51],  # white
        ]
    )

    def colour_thruster_texture(colour):
        return THRUSTER_TEXTURE_16_RGBA.at[:9, :, :3].mul(colour[None, None, :] / 255.0)

    coloured_thruster_textures = jax.vmap(colour_thruster_texture)(JOINT_COLOURS)

    ROLE_COLOURS = jnp.array(
        [
            [160.0, 160.0, 160.0],  # None
            [0.0, 204.0, 0.0],  # Green:    The ball
            [0.0, 102.0, 204.0],  # Blue:   The goal
            [255.0, 102.0, 102.0],  # Red:      Death Objects
        ]
    )

    BACKGROUND_COLOUR = jnp.array([255.0, 255.0, 255.0])

    def _get_colour(shape_role, inverse_inertia):
        base_colour = ROLE_COLOURS[shape_role]
        f = (inverse_inertia == 0) * 1
        is_not_normal = (shape_role != 0) * 1

        return jnp.array(
            [
                base_colour,
                base_colour,
                FIXATED_COLOUR,
                base_colour * 0.5,
            ]
        )[2 * f + is_not_normal]

    # Pixels per unit distance
    ppud = env_params.pixels_per_unit // downscale

    downscaled_screen_dim = (screen_dim[0] // downscale, screen_dim[1] // downscale)

    full_screen_size = (
        downscaled_screen_dim[0] + (static_params.max_shape_size * 2 * ppud),
        downscaled_screen_dim[1] + (static_params.max_shape_size * 2 * ppud),
    )
    cleared_screen = clear_screen(full_screen_size, BACKGROUND_COLOUR)

    def _world_space_to_pixel_space(x):
        return (x + static_params.max_shape_size) * ppud

    def fragment_shader_kinetix_circle(position, current_frag, unit_position, uniform):
        centre, radius, rotation, colour, mask = uniform

        dist = jnp.sqrt(jnp.square(position - centre).sum())
        inside = dist <= radius
        on_edge = dist > radius - edge_thickness

        normal = jnp.array([jnp.sin(rotation), -jnp.cos(rotation)])

        dist = dist_from_line(position, centre, centre + normal)

        on_edge |= (dist < 1) & (jnp.dot(normal, position - centre) <= 0)

        fragment = jax.lax.select(on_edge, jnp.zeros(3), colour)

        return jax.lax.select(inside & mask, fragment, current_frag)

    def fragment_shader_kinetix_joint_circle(position, current_frag, unit_position, uniform):
        centre, radius, colour, mask = uniform
        new = fragment_shader_circle(position, current_frag, unit_position, (centre, radius, colour))
        return new * mask + (1 - mask) * current_frag

    def fragment_shader_kinetix_joint_texture(position, current_frag, unit_position, uniform):
        texture, colour, mask = uniform

        tex_coord = (
            jnp.array(
                [
                    joint_tex_size * unit_position[0],
                    joint_tex_size * unit_position[1],
                ]
            )
            - 0.5
        )

        tex_frag = nearest_neighbour(texture, tex_coord)
        tex_frag = tex_frag.at[3].mul(mask)
        tex_frag = tex_frag.at[:3].mul(colour / 255.0)

        tex_frag = (tex_frag[3] * tex_frag[:3]) + ((1.0 - tex_frag[3]) * current_frag)

        return tex_frag

    thruster_pixel_size = thruster_tex_size // downscale
    thruster_pixel_size_diagonal = (thruster_pixel_size * np.sqrt(2)).astype(jnp.int32) + 1

    def fragment_shader_kinetix_thruster(fragment_position, current_frag, unit_position, uniform):
        thruster_position, rotation, texture, mask = uniform

        tex_position = jnp.matmul(rmat(-rotation), (fragment_position - thruster_position)) / thruster_pixel_size + 0.5

        mask &= (tex_position[0] >= 0) & (tex_position[0] <= 1) & (tex_position[1] >= 0) & (tex_position[1] <= 1)

        eps = 0.001
        tex_coord = (
            jnp.array(
                [
                    thruster_tex_size * tex_position[0],
                    thruster_tex_size * tex_position[1],
                ]
            )
            - 0.5
            + eps
        )

        tex_frag = nearest_neighbour(texture, tex_coord)
        tex_frag = tex_frag.at[3].mul(mask)

        tex_frag = (tex_frag[3] * tex_frag[:3]) + ((1.0 - tex_frag[3]) * current_frag)

        return tex_frag

    patch_size_1d = static_params.max_shape_size * ppud
    patch_size = (patch_size_1d, patch_size_1d)

    circle_renderer = make_renderer(full_screen_size, fragment_shader_kinetix_circle, patch_size, batched=True)
    quad_renderer = make_renderer(full_screen_size, fragment_shader_quad_no_edges, patch_size, batched=True)
    big_quad_renderer = make_renderer(full_screen_size, fragment_shader_quad_no_edges, downscaled_screen_dim)

    joint_pixel_size = joint_tex_size // downscale
    joint_fragment_shader = (
        fragment_shader_kinetix_joint_texture if use_textures_for_joints else fragment_shader_kinetix_joint_circle
    )
    joint_renderer = make_renderer(
        full_screen_size, joint_fragment_shader, (joint_pixel_size, joint_pixel_size), batched=True
    )

    thruster_renderer = make_renderer(
        full_screen_size,
        fragment_shader_kinetix_thruster,
        (thruster_pixel_size_diagonal, thruster_pixel_size_diagonal),
        batched=True,
    )

    @jax.jit
    def render_pixels(state: EnvState):
        pixels = cleared_screen

        # Floor
        floor_uniform = (
            _world_space_to_pixel_space(state.polygon.position[0, None, :] + state.polygon.vertices[0]),
            _get_colour(state.polygon_shape_roles[0], 0),
            jnp.zeros(3),
            True,
        )

        pixels = big_quad_renderer(pixels, _world_space_to_pixel_space(jnp.zeros(2, dtype=jnp.int32)), floor_uniform)

        # Rectangles
        rectangle_patch_positions = _world_space_to_pixel_space(
            state.polygon.position - (static_params.max_shape_size / 2.0)
        ).astype(jnp.int32)

        rectangle_rmats = jax.vmap(rmat)(state.polygon.rotation)
        rectangle_rmats = jnp.repeat(rectangle_rmats[:, None, :, :], repeats=static_params.max_polygon_vertices, axis=1)
        rectangle_vertices_pixel_space = _world_space_to_pixel_space(
            state.polygon.position[:, None, :] + jax.vmap(jax.vmap(jnp.matmul))(rectangle_rmats, state.polygon.vertices)
        )
        rectangle_colours = jax.vmap(_get_colour)(state.polygon_shape_roles, state.polygon.inverse_mass)
        rectangle_edge_colours = jnp.zeros((static_params.num_polygons, 3))

        rectangle_uniforms = (
            rectangle_vertices_pixel_space,
            rectangle_colours,
            rectangle_edge_colours,
            state.polygon.active.at[: static_params.num_static_fixated_polys].set(False),
        )

        pixels = quad_renderer(pixels, rectangle_patch_positions, rectangle_uniforms)

        # Circles
        circle_positions_pixel_space = _world_space_to_pixel_space(state.circle.position)
        circle_radii_pixel_space = state.circle.radius * ppud
        circle_patch_positions = _world_space_to_pixel_space(
            state.circle.position - (static_params.max_shape_size / 2.0)
        ).astype(jnp.int32)

        circle_colours = jax.vmap(_get_colour)(state.circle_shape_roles, state.circle.inverse_mass)

        circle_uniforms = (
            circle_positions_pixel_space,
            circle_radii_pixel_space,
            state.circle.rotation,
            circle_colours,
            state.circle.active,
        )

        pixels = circle_renderer(pixels, circle_patch_positions, circle_uniforms)

        # Joints
        joint_patch_positions = jnp.round(
            _world_space_to_pixel_space(state.joint.global_position) - (joint_pixel_size // 2)
        ).astype(jnp.int32)
        joint_textures = jax.vmap(jax.lax.select, in_axes=(0, None, None))(
            state.joint.is_fixed_joint, FJOINT_TEXTURE_6_RGBA, RJOINT_TEXTURE_6_RGBA
        )
        joint_colours = JOINT_COLOURS[
            (state.motor_bindings + 1) * (state.joint.motor_on & (~state.joint.is_fixed_joint))
        ]

        if use_textures_for_joints:
            joint_uniforms = (joint_textures, joint_colours, state.joint.active)
        else:
            joint_uniforms = (
                joint_patch_positions + joint_pixel_size // 2,
                jnp.ones(static_params.num_joints) * joint_pixel_size / 2,
                joint_colours.astype(jnp.float32),
                state.joint.active,
            )

        pixels = joint_renderer(pixels, joint_patch_positions, joint_uniforms)

        # Thrusters
        thruster_positions = jnp.round(_world_space_to_pixel_space(state.thruster.global_position)).astype(jnp.int32)
        thruster_patch_positions = thruster_positions - (thruster_pixel_size_diagonal // 2)
        thruster_textures = coloured_thruster_textures[state.thruster_bindings + 1]
        thruster_rotations = (
            state.thruster.rotation
            + jax.vmap(select_shape, in_axes=(None, 0, None))(
                state, state.thruster.object_index, static_params
            ).rotation
        )
        thruster_uniforms = (thruster_positions, thruster_rotations, thruster_textures, state.thruster.active)

        pixels = thruster_renderer(pixels, thruster_patch_positions, thruster_uniforms)

        # Crop out the sides
        crop_amount = static_params.max_shape_size * ppud
        return pixels[crop_amount:-crop_amount, crop_amount:-crop_amount]

    return render_pixels


@struct.dataclass
class PixelsObservation:
    image: jnp.ndarray
    global_info: jnp.ndarray


def make_render_pixels_rl(env_params, static_params: StaticEnvParams):
    render_fn = make_render_pixels(env_params, static_params)

    def inner(state):
        pixels = render_fn(state) / 255.0
        return PixelsObservation(
            image=pixels,
            global_info=jnp.array([state.gravity[1] / 10.0]),
        )

    return inner

def make_render_pixels_unanno( # Renamed for clarity if you keep original
    env_params,
    static_params: StaticEnvParams,
):
    magnification_factor = 4

    # Ensure original parameters used for setup are Python native types
    original_screen_dim = tuple(map(int, static_params.screen_dim))
    original_downscale = int(static_params.downscale / magnification_factor)
    original_pixels_per_unit = float(env_params.pixels_per_unit)
    py_max_shape_size = float(static_params.max_shape_size) * magnification_factor

    joint_tex_size = 6
    thruster_tex_size = 16

    FIXATED_COLOUR = jnp.array([80, 80, 80])
    JOINT_COLOURS = jnp.array(
        [[255, 255, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255], [255, 153, 51]]
    )

    def colour_thruster_texture(colour):
        return THRUSTER_TEXTURE_16_RGBA.at[:9, :, :3].mul(colour[None, None, :] / 255.0)
    coloured_thruster_textures = jax.vmap(colour_thruster_texture)(JOINT_COLOURS)

    ROLE_COLOURS = jnp.array(
        [[160.0, 160.0, 160.0], [0.0, 204.0, 0.0], [0.0, 102.0, 204.0], [255.0, 102.0, 102.0]]
    )
    BACKGROUND_COLOUR = jnp.array([255.0, 255.0, 255.0])

    def _get_colour(shape_role, inverse_inertia):
        base_colour = ROLE_COLOURS[shape_role]
        f = (inverse_inertia == 0) * 1
        is_not_normal = (shape_role != 0) * 1
        return jnp.array(
            [base_colour, base_colour, FIXATED_COLOUR, base_colour * 0.5]
        )[2 * f + is_not_normal]

    # --- Calculate dimensions as Python floats/ints for setup ---
    # This is the effective pixels per unit distance *as a float* for precise calcs
    effective_ppud_float = (original_pixels_per_unit * magnification_factor) / original_downscale

    # Downscaled screen dimensions (Python ints)
    py_downscaled_screen_dim_0 = int(round((original_screen_dim[0]) / original_downscale) * int(magnification_factor))
    py_downscaled_screen_dim_1 = int(round((original_screen_dim[1]) / original_downscale) * int(magnification_factor))
    py_downscaled_screen_dim = (py_downscaled_screen_dim_0, py_downscaled_screen_dim_1)

    # Padding for shapes that might go off-screen (Python int)
    # py_max_shape_size * 2 is total padding in world units. Multiplied by effective_ppud_float gives pixel padding.
    py_full_screen_size_padding = int(round(py_max_shape_size * 2 * effective_ppud_float))

    # Full screen size including padding (Python ints)
    py_full_screen_size_0 = py_downscaled_screen_dim_0 + py_full_screen_size_padding
    py_full_screen_size_1 = py_downscaled_screen_dim_1 + py_full_screen_size_padding
    py_full_screen_size = (py_full_screen_size_0, py_full_screen_size_1)

    print(f"Full screen size: {py_full_screen_size},Downscaled screen dim: {py_downscaled_screen_dim}, py_full_screen_size_padding: {py_full_screen_size_padding}")

    # This is now a tuple of Python ints, suitable for jnp.ones or static args
    cleared_screen = clear_screen(py_full_screen_size, BACKGROUND_COLOUR) # LINE 350 in original trace

    # This function will be used inside the JITted `render_pixels`
    # It uses the float `effective_ppud_float` for precision with JAX array inputs
    def _world_space_to_pixel_space(x): # x is JAX array
        return (x + py_max_shape_size) * effective_ppud_float

    # --- Fragment Shaders (definitions remain mostly the same) ---
    def fragment_shader_kinetix_circle(position, current_frag, unit_position, uniform):
        centre, radius, rotation, colour, mask = uniform
        dist = jnp.sqrt(jnp.square(position - centre).sum())
        inside = dist <= radius
        scaled_edge_pixel_width = max(1.0, 2.0 * magnification_factor)
        on_edge = dist > (radius - scaled_edge_pixel_width)
        normal = jnp.array([jnp.sin(rotation), -jnp.cos(rotation)])
        # Simplified notch logic from before
        delta_pos = position - centre
        angle_pixel = jnp.arctan2(delta_pos[1], delta_pos[0])
        target_angle = rotation - jnp.pi / 2.0
        angle_diff = jnp.fmod(angle_pixel - target_angle + jnp.pi, 2.0 * jnp.pi) - jnp.pi
        angular_width = (0.05 / magnification_factor)
        on_notch_line = (jnp.abs(angle_diff) < angular_width) & (jnp.dot(normal, position - centre) <= 0)
        on_edge |= on_notch_line
        fragment = jax.lax.select(on_edge, jnp.zeros(3), colour)
        return jax.lax.select(inside & mask, fragment, current_frag)

    def fragment_shader_kinetix_joint(position, current_frag, unit_position, uniform):
        texture, colour, mask = uniform
        tex_coord = (jnp.array([joint_tex_size * unit_position[0], joint_tex_size * unit_position[1]]) - 0.5)
        tex_frag = nearest_neighbour(texture, tex_coord)
        tex_frag = tex_frag.at[3].mul(mask)
        tex_frag = tex_frag.at[:3].mul(colour / 255.0)
        tex_frag = (tex_frag[3] * tex_frag[:3]) + ((1.0 - tex_frag[3]) * current_frag)
        return tex_frag

    # Thruster pixel size for shader transformation (Python float)
    py_thruster_pixel_size_on_screen_float = (thruster_tex_size / original_downscale) * magnification_factor
    # Ensure it's at least a small positive number for division
    safe_py_thruster_pixel_size_on_screen_float = max(1e-6, py_thruster_pixel_size_on_screen_float)


    def fragment_shader_kinetix_thruster(fragment_position, current_frag, unit_position, uniform):
        thruster_position, rotation, texture, mask = uniform
        # Use the Python float for scaling in transformation
        tex_position = jnp.matmul(rmat(-rotation), (fragment_position - thruster_position)) / safe_py_thruster_pixel_size_on_screen_float + 0.5
        mask &= (tex_position[0] >= 0) & (tex_position[0] <= 1) & (tex_position[1] >= 0) & (tex_position[1] <= 1)
        eps = 0.001
        tex_coord = (jnp.array([thruster_tex_size * tex_position[0], thruster_tex_size * tex_position[1]]) - 0.5 + eps)
        tex_frag = nearest_neighbour(texture, tex_coord)
        tex_frag = tex_frag.at[3].mul(mask)
        tex_frag = (tex_frag[3] * tex_frag[:3]) + ((1.0 - tex_frag[3]) * current_frag)
        return tex_frag

    # --- Calculate patch sizes as Python ints for renderer setup ---
    py_patch_size_1d_float = py_max_shape_size * effective_ppud_float
    py_patch_size_1d = max(1, int(round(py_patch_size_1d_float)))
    py_patch_size = (py_patch_size_1d, py_patch_size_1d) # Tuple of Python ints

    # Joint pixel size for patch (Python ints)
    py_joint_pixel_size_on_screen_float = (joint_tex_size / original_downscale) * magnification_factor
    py_joint_pixel_size_on_screen = max(1, int(round(py_joint_pixel_size_on_screen_float)))
    py_joint_patch_size = (py_joint_pixel_size_on_screen, py_joint_pixel_size_on_screen)

    # Thruster pixel size for patch (Python ints)
    py_thruster_pixel_size_on_screen = max(1, int(round(py_thruster_pixel_size_on_screen_float)))
    py_thruster_pixel_size_diagonal = max(
        py_thruster_pixel_size_on_screen,
        int(round(py_thruster_pixel_size_on_screen * np.sqrt(2.0))) + 1
    )
    py_thruster_patch_size = (py_thruster_pixel_size_diagonal, py_thruster_pixel_size_diagonal)

    # --- Setup Renderers with Python int dimensions for static parts ---
    circle_renderer = make_renderer(py_full_screen_size, fragment_shader_kinetix_circle, py_patch_size, batched=True)
    quad_renderer = make_renderer(py_full_screen_size, fragment_shader_edged_quad, py_patch_size, batched=True)
    # py_downscaled_screen_dim is already a tuple of Python ints
    big_quad_renderer = make_renderer(py_full_screen_size, fragment_shader_edged_quad, py_downscaled_screen_dim)
    joint_renderer = make_renderer(py_full_screen_size, fragment_shader_kinetix_joint, py_joint_patch_size, batched=True)
    thruster_renderer = make_renderer(py_full_screen_size, fragment_shader_kinetix_thruster, py_thruster_patch_size, batched=True)

    @jax.jit
    def render_pixels(state: EnvState):
        pixels = cleared_screen # This is a JAX array initialized correctly

        # Floor
        floor_vertices_px = _world_space_to_pixel_space(
            state.polygon.position[0, None, :] + state.polygon.vertices[0]
        )
        floor_uniform = (
            floor_vertices_px,
            _get_colour(state.polygon_shape_roles[0], 0),
            jnp.zeros(3), True,
        )
        floor_patch_pos = _world_space_to_pixel_space(jnp.zeros(2, dtype=jnp.float32)).astype(jnp.int32)
        pixels = big_quad_renderer(pixels, floor_patch_pos, floor_uniform)

        # Rectangles
        rectangle_patch_positions = _world_space_to_pixel_space(
            state.polygon.position - (py_max_shape_size / 2.0)
        ).astype(jnp.int32)
        rectangle_rmats = jax.vmap(rmat)(state.polygon.rotation)
        rectangle_rmats = jnp.repeat(rectangle_rmats[:, None, :, :], repeats=static_params.max_polygon_vertices, axis=1)
        rectangle_vertices_pixel_space = _world_space_to_pixel_space(
            state.polygon.position[:, None, :] + jax.vmap(jax.vmap(jnp.matmul))(rectangle_rmats, state.polygon.vertices)
        )
        rectangle_colours = jax.vmap(_get_colour)(state.polygon_shape_roles, state.polygon.inverse_mass)
        rectangle_edge_colours = jnp.zeros((static_params.num_polygons, 3))
        rectangle_uniforms = (
            rectangle_vertices_pixel_space, rectangle_colours, rectangle_edge_colours,
            state.polygon.active.at[: static_params.num_static_fixated_polys].set(False),
        )
        pixels = quad_renderer(pixels, rectangle_patch_positions, rectangle_uniforms)

        # Circles
        circle_positions_pixel_space = _world_space_to_pixel_space(state.circle.position)
        # state.circle.radius (JAX array) * effective_ppud_float (Python float) -> JAX array
        circle_radii_pixel_space = state.circle.radius * effective_ppud_float
        circle_patch_positions = _world_space_to_pixel_space(
            state.circle.position - (py_max_shape_size / 2.0)
        ).astype(jnp.int32)
        circle_colours = jax.vmap(_get_colour)(state.circle_shape_roles, state.circle.inverse_mass)
        circle_uniforms = (
            circle_positions_pixel_space, circle_radii_pixel_space, state.circle.rotation,
            circle_colours, state.circle.active,
        )
        pixels = circle_renderer(pixels, circle_patch_positions, circle_uniforms)

        # Joints
        # py_joint_pixel_size_on_screen_float is Python float
        joint_patch_positions = jnp.round(
            _world_space_to_pixel_space(state.joint.global_position) - (py_joint_pixel_size_on_screen_float / 2.0)
        ).astype(jnp.int32)
        joint_textures = jax.vmap(jax.lax.select, in_axes=(0, None, None))(
            state.joint.is_fixed_joint, FJOINT_TEXTURE_6_RGBA, RJOINT_TEXTURE_6_RGBA
        )
        joint_colours = JOINT_COLOURS[
            (state.motor_bindings + 1) * (state.joint.motor_on & (~state.joint.is_fixed_joint))
        ]
        joint_uniforms = (joint_textures, joint_colours, state.joint.active)
        pixels = joint_renderer(pixels, joint_patch_positions, joint_uniforms)

        # Thrusters
        thruster_positions_px_float = _world_space_to_pixel_space(state.thruster.global_position) # JAX array, float
        # py_thruster_pixel_size_diagonal is Python int, convert to float for division
        thruster_patch_positions = jnp.round(
            thruster_positions_px_float - (float(py_thruster_pixel_size_diagonal) / 2.0)
        ).astype(jnp.int32)
        thruster_textures = coloured_thruster_textures[state.thruster_bindings + 1]
        thruster_rotations = (
            state.thruster.rotation +
            jax.vmap(select_shape, in_axes=(None, 0, None))(state, state.thruster.object_index, static_params).rotation
        )
        # Shader expects thruster center positions as int, if so:
        thruster_uniforms = (
            jnp.round(thruster_positions_px_float).astype(jnp.int32), # Or pass thruster_positions_px_float if shader handles floats
            thruster_rotations, thruster_textures, state.thruster.active
        )
        pixels = thruster_renderer(pixels, thruster_patch_positions, thruster_uniforms)

        # Crop out the sides
        # py_full_screen_size_padding is Python int
        py_crop_amount = py_full_screen_size_padding // 2

        final_h, final_w = pixels.shape[0], pixels.shape[1] # These are Python ints from JAX array.shape
        # Ensure crop doesn't go out of bounds, slices must be Python ints or JAX scalars.
        # py_crop_amount is already Python int.
        crop_top = min(py_crop_amount, final_h // 2)
        crop_left = min(py_crop_amount, final_w // 2)

        # Slicing with python integers is fine.
        return pixels[crop_top : final_h - crop_top, crop_left : final_w - crop_left]

    return render_pixels

def make_render_pixels_anno( # Renamed for clarity if you keep original
    env_params,
    static_params: StaticEnvParams,
):
    magnification_factor = 4

    # Ensure original parameters used for setup are Python native types
    original_screen_dim = tuple(map(int, static_params.screen_dim))
    original_downscale = int(static_params.downscale / magnification_factor)
    original_pixels_per_unit = float(env_params.pixels_per_unit)
    py_max_shape_size = float(static_params.max_shape_size) * magnification_factor

    joint_tex_size = 6
    thruster_tex_size = 16

    FIXATED_COLOUR = jnp.array([80, 80, 80])
    JOINT_COLOURS = jnp.array(
        [[255, 255, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255], [255, 153, 51]]
    )

    def colour_thruster_texture(colour):
        return THRUSTER_TEXTURE_16_RGBA.at[:9, :, :3].mul(colour[None, None, :] / 255.0)
    coloured_thruster_textures = jax.vmap(colour_thruster_texture)(JOINT_COLOURS)

    ROLE_COLOURS = jnp.array(
        [[160.0, 160.0, 160.0], [0.0, 204.0, 0.0], [0.0, 102.0, 204.0], [255.0, 102.0, 102.0]]
    )
    BACKGROUND_COLOUR = jnp.array([255.0, 255.0, 255.0])

    def _get_colour(shape_role, inverse_inertia):
        base_colour = ROLE_COLOURS[shape_role]
        f = (inverse_inertia == 0) * 1
        is_not_normal = (shape_role != 0) * 1
        return jnp.array(
            [base_colour, base_colour, FIXATED_COLOUR, base_colour * 0.5]
        )[2 * f + is_not_normal]

    # --- Calculate dimensions as Python floats/ints for setup ---
    # This is the effective pixels per unit distance *as a float* for precise calcs
    effective_ppud_float = (original_pixels_per_unit * magnification_factor) / original_downscale

    # Downscaled screen dimensions (Python ints)
    py_downscaled_screen_dim_0 = int(round((original_screen_dim[0]) / original_downscale) * int(magnification_factor))
    py_downscaled_screen_dim_1 = int(round((original_screen_dim[1]) / original_downscale) * int(magnification_factor))
    py_downscaled_screen_dim = (py_downscaled_screen_dim_0, py_downscaled_screen_dim_1)

    # Padding for shapes that might go off-screen (Python int)
    # py_max_shape_size * 2 is total padding in world units. Multiplied by effective_ppud_float gives pixel padding.
    py_full_screen_size_padding = int(round(py_max_shape_size * 2 * effective_ppud_float))

    # Full screen size including padding (Python ints)
    py_full_screen_size_0 = py_downscaled_screen_dim_0 + py_full_screen_size_padding
    py_full_screen_size_1 = py_downscaled_screen_dim_1 + py_full_screen_size_padding
    py_full_screen_size = (py_full_screen_size_0, py_full_screen_size_1)

    print(f"Full screen size: {py_full_screen_size},Downscaled screen dim: {py_downscaled_screen_dim}, py_full_screen_size_padding: {py_full_screen_size_padding}")

    # This is now a tuple of Python ints, suitable for jnp.ones or static args
    cleared_screen = clear_screen(py_full_screen_size, BACKGROUND_COLOUR) # LINE 350 in original trace

    # This function will be used inside the JITted `render_pixels`
    # It uses the float `effective_ppud_float` for precision with JAX array inputs
    def _world_space_to_pixel_space(x): # x is JAX array
        return (x + py_max_shape_size) * effective_ppud_float

    # --- Fragment Shaders (definitions remain mostly the same) ---
    def fragment_shader_kinetix_circle(position, current_frag, unit_position, uniform):
        centre, radius, rotation, colour, mask = uniform
        dist = jnp.sqrt(jnp.square(position - centre).sum())
        inside = dist <= radius
        scaled_edge_pixel_width = max(1.0, 2.0 * magnification_factor)
        on_edge = dist > (radius - scaled_edge_pixel_width)
        normal = jnp.array([jnp.sin(rotation), -jnp.cos(rotation)])
        # Simplified notch logic from before
        delta_pos = position - centre
        angle_pixel = jnp.arctan2(delta_pos[1], delta_pos[0])
        target_angle = rotation - jnp.pi / 2.0
        angle_diff = jnp.fmod(angle_pixel - target_angle + jnp.pi, 2.0 * jnp.pi) - jnp.pi
        angular_width = (0.05 / magnification_factor)
        on_notch_line = (jnp.abs(angle_diff) < angular_width) & (jnp.dot(normal, position - centre) <= 0)
        on_edge |= on_notch_line
        fragment = jax.lax.select(on_edge, jnp.zeros(3), colour)
        return jax.lax.select(inside & mask, fragment, current_frag)

    def fragment_shader_kinetix_joint(position, current_frag, unit_position, uniform):
        texture, colour, mask = uniform
        tex_coord = (jnp.array([joint_tex_size * unit_position[0], joint_tex_size * unit_position[1]]) - 0.5)
        tex_frag = nearest_neighbour(texture, tex_coord)
        tex_frag = tex_frag.at[3].mul(mask)
        tex_frag = tex_frag.at[:3].mul(colour / 255.0)
        tex_frag = (tex_frag[3] * tex_frag[:3]) + ((1.0 - tex_frag[3]) * current_frag)
        return tex_frag

    # Thruster pixel size for shader transformation (Python float)
    py_thruster_pixel_size_on_screen_float = (thruster_tex_size / original_downscale) * magnification_factor
    # Ensure it's at least a small positive number for division
    safe_py_thruster_pixel_size_on_screen_float = max(1e-6, py_thruster_pixel_size_on_screen_float)


    def fragment_shader_kinetix_thruster(fragment_position, current_frag, unit_position, uniform):
        thruster_position, rotation, texture, mask = uniform
        # Use the Python float for scaling in transformation
        tex_position = jnp.matmul(rmat(-rotation), (fragment_position - thruster_position)) / safe_py_thruster_pixel_size_on_screen_float + 0.5
        mask &= (tex_position[0] >= 0) & (tex_position[0] <= 1) & (tex_position[1] >= 0) & (tex_position[1] <= 1)
        eps = 0.001
        tex_coord = (jnp.array([thruster_tex_size * tex_position[0], thruster_tex_size * tex_position[1]]) - 0.5 + eps)
        tex_frag = nearest_neighbour(texture, tex_coord)
        tex_frag = tex_frag.at[3].mul(mask)
        tex_frag = (tex_frag[3] * tex_frag[:3]) + ((1.0 - tex_frag[3]) * current_frag)
        return tex_frag

    # --- Calculate patch sizes as Python ints for renderer setup ---
    py_patch_size_1d_float = py_max_shape_size * effective_ppud_float
    py_patch_size_1d = max(1, int(round(py_patch_size_1d_float)))
    py_patch_size = (py_patch_size_1d, py_patch_size_1d) # Tuple of Python ints

    # Joint pixel size for patch (Python ints)
    py_joint_pixel_size_on_screen_float = (joint_tex_size / original_downscale) * magnification_factor
    py_joint_pixel_size_on_screen = max(1, int(round(py_joint_pixel_size_on_screen_float)))
    py_joint_patch_size = (py_joint_pixel_size_on_screen, py_joint_pixel_size_on_screen)

    # Thruster pixel size for patch (Python ints)
    py_thruster_pixel_size_on_screen = max(1, int(round(py_thruster_pixel_size_on_screen_float)))
    py_thruster_pixel_size_diagonal = max(
        py_thruster_pixel_size_on_screen,
        int(round(py_thruster_pixel_size_on_screen * np.sqrt(2.0))) + 1
    )
    py_thruster_patch_size = (py_thruster_pixel_size_diagonal, py_thruster_pixel_size_diagonal)

    # --- Setup Renderers with Python int dimensions for static parts ---
    circle_renderer = make_renderer(py_full_screen_size, fragment_shader_kinetix_circle, py_patch_size, batched=True)
    quad_renderer = make_renderer(py_full_screen_size, fragment_shader_edged_quad, py_patch_size, batched=True)
    # py_downscaled_screen_dim is already a tuple of Python ints
    big_quad_renderer = make_renderer(py_full_screen_size, fragment_shader_edged_quad, py_downscaled_screen_dim)
    joint_renderer = make_renderer(py_full_screen_size, fragment_shader_kinetix_joint, py_joint_patch_size, batched=True)
    thruster_renderer = make_renderer(py_full_screen_size, fragment_shader_kinetix_thruster, py_thruster_patch_size, batched=True)

    # @jax.jit
    def render_pixels(state: EnvState):
        pixels = cleared_screen # This is a JAX array initialized correctly

        # Floor
        floor_vertices_px = _world_space_to_pixel_space(
            state.polygon.position[0, None, :] + state.polygon.vertices[0]
        )
        floor_uniform = (
            floor_vertices_px,
            _get_colour(state.polygon_shape_roles[0], 0),
            jnp.zeros(3), True,
        )
        floor_patch_pos = _world_space_to_pixel_space(jnp.zeros(2, dtype=jnp.float32)).astype(jnp.int32)
        pixels = big_quad_renderer(pixels, floor_patch_pos, floor_uniform)

        # Rectangles
        rectangle_patch_positions = _world_space_to_pixel_space(
            state.polygon.position - (py_max_shape_size / 2.0)
        ).astype(jnp.int32)
        rectangle_rmats = jax.vmap(rmat)(state.polygon.rotation)
        rectangle_rmats = jnp.repeat(rectangle_rmats[:, None, :, :], repeats=static_params.max_polygon_vertices, axis=1)
        rectangle_vertices_pixel_space = _world_space_to_pixel_space(
            state.polygon.position[:, None, :] + jax.vmap(jax.vmap(jnp.matmul))(rectangle_rmats, state.polygon.vertices)
        )
        rectangle_colours = jax.vmap(_get_colour)(state.polygon_shape_roles, state.polygon.inverse_mass)
        rectangle_edge_colours = jnp.zeros((static_params.num_polygons, 3))
        rectangle_uniforms = (
            rectangle_vertices_pixel_space, rectangle_colours, rectangle_edge_colours,
            state.polygon.active.at[: static_params.num_static_fixated_polys].set(False),
        )
        pixels = quad_renderer(pixels, rectangle_patch_positions, rectangle_uniforms)

        # Circles
        circle_positions_pixel_space = _world_space_to_pixel_space(state.circle.position)
        # state.circle.radius (JAX array) * effective_ppud_float (Python float) -> JAX array
        circle_radii_pixel_space = state.circle.radius * effective_ppud_float
        circle_patch_positions = _world_space_to_pixel_space(
            state.circle.position - (py_max_shape_size / 2.0)
        ).astype(jnp.int32)
        circle_colours = jax.vmap(_get_colour)(state.circle_shape_roles, state.circle.inverse_mass)
        circle_uniforms = (
            circle_positions_pixel_space, circle_radii_pixel_space, state.circle.rotation,
            circle_colours, state.circle.active,
        )
        pixels = circle_renderer(pixels, circle_patch_positions, circle_uniforms)

        # Joints
        # py_joint_pixel_size_on_screen_float is Python float
        joint_patch_positions = jnp.round(
            _world_space_to_pixel_space(state.joint.global_position) - (py_joint_pixel_size_on_screen_float / 2.0)
        ).astype(jnp.int32)
        joint_textures = jax.vmap(jax.lax.select, in_axes=(0, None, None))(
            state.joint.is_fixed_joint, FJOINT_TEXTURE_6_RGBA, RJOINT_TEXTURE_6_RGBA
        )
        joint_colours = JOINT_COLOURS[
            (state.motor_bindings + 1) * (state.joint.motor_on & (~state.joint.is_fixed_joint))
        ]
        joint_uniforms = (joint_textures, joint_colours, state.joint.active)
        pixels = joint_renderer(pixels, joint_patch_positions, joint_uniforms)

        # Thrusters
        thruster_positions_px_float = _world_space_to_pixel_space(state.thruster.global_position) # JAX array, float
        # py_thruster_pixel_size_diagonal is Python int, convert to float for division
        thruster_patch_positions = jnp.round(
            thruster_positions_px_float - (float(py_thruster_pixel_size_diagonal) / 2.0)
        ).astype(jnp.int32)
        thruster_textures = coloured_thruster_textures[state.thruster_bindings + 1]
        thruster_rotations = (
            state.thruster.rotation +
            jax.vmap(select_shape, in_axes=(None, 0, None))(state, state.thruster.object_index, static_params).rotation
        )
        # Shader expects thruster center positions as int, if so:
        thruster_uniforms = (
            jnp.round(thruster_positions_px_float).astype(jnp.int32), # Or pass thruster_positions_px_float if shader handles floats
            thruster_rotations, thruster_textures, state.thruster.active
        )
        pixels = thruster_renderer(pixels, thruster_patch_positions, thruster_uniforms)

        # Crop out the sides
        # py_full_screen_size_padding is Python int
        py_crop_amount = py_full_screen_size_padding // 2

        final_h, final_w = pixels.shape[0], pixels.shape[1] # These are Python ints from JAX array.shape
        # Ensure crop doesn't go out of bounds, slices must be Python ints or JAX scalars.
        # py_crop_amount is already Python int.
        crop_top = min(py_crop_amount, final_h // 2)
        crop_left = min(py_crop_amount, final_w // 2)

        # Slicing with python integers is fine.
        return pixels[crop_top : final_h - crop_top, crop_left : final_w - crop_left]

    @jax.jit
    def render_pixels_and_coords(state: EnvState): # 新函数名，或者覆盖旧的
        pixels = cleared_screen

        # --- 地板和矩形渲染 (保持不变) ---
        # Floor
        floor_vertices_px = _world_space_to_pixel_space(
            state.polygon.position[0, None, :] + state.polygon.vertices[0]
        )
        floor_uniform = (
            floor_vertices_px,
            _get_colour(state.polygon_shape_roles[0], 0),
            jnp.zeros(3), True,
        )
        floor_patch_pos = _world_space_to_pixel_space(jnp.zeros(2, dtype=jnp.float32)).astype(jnp.int32)
        pixels = big_quad_renderer(pixels, floor_patch_pos, floor_uniform)

        # Rectangles
        rectangle_patch_positions = _world_space_to_pixel_space(
            state.polygon.position - (py_max_shape_size / 2.0)
        ).astype(jnp.int32)
        rectangle_rmats = jax.vmap(rmat)(state.polygon.rotation)
        rectangle_rmats = jnp.repeat(rectangle_rmats[:, None, :, :], repeats=static_params.max_polygon_vertices, axis=1)
        rectangle_vertices_pixel_space = _world_space_to_pixel_space(
            state.polygon.position[:, None, :] + jax.vmap(jax.vmap(jnp.matmul))(rectangle_rmats, state.polygon.vertices)
        )
        rectangle_colours = jax.vmap(_get_colour)(state.polygon_shape_roles, state.polygon.inverse_mass)
        rectangle_edge_colours = jnp.zeros((static_params.num_polygons, 3))
        rectangle_uniforms = (
            rectangle_vertices_pixel_space, rectangle_colours, rectangle_edge_colours,
            state.polygon.active.at[: static_params.num_static_fixated_polys].set(False),
        )
        pixels = quad_renderer(pixels, rectangle_patch_positions, rectangle_uniforms)

        # Circles
        circle_positions_pixel_space = _world_space_to_pixel_space(state.circle.position)
        # state.circle.radius (JAX array) * effective_ppud_float (Python float) -> JAX array
        circle_radii_pixel_space = state.circle.radius * effective_ppud_float
        circle_patch_positions = _world_space_to_pixel_space(
            state.circle.position - (py_max_shape_size / 2.0)
        ).astype(jnp.int32)
        circle_colours = jax.vmap(_get_colour)(state.circle_shape_roles, state.circle.inverse_mass)
        circle_uniforms = (
            circle_positions_pixel_space, circle_radii_pixel_space, state.circle.rotation,
            circle_colours, state.circle.active,
        )
        pixels = circle_renderer(pixels, circle_patch_positions, circle_uniforms)

        # --- Joints渲染 (几乎不变, 但我们保存坐标) ---
        joint_positions_px_float = _world_space_to_pixel_space(state.joint.global_position) # 保存这个
        joint_patch_positions = jnp.round(
            joint_positions_px_float - (py_joint_pixel_size_on_screen_float / 2.0)
        ).astype(jnp.int32)
        joint_textures = jax.vmap(jax.lax.select, in_axes=(0, None, None))(
            state.joint.is_fixed_joint, FJOINT_TEXTURE_6_RGBA, RJOINT_TEXTURE_6_RGBA
        )
        joint_colours = JOINT_COLOURS[
            (state.motor_bindings + 1) * (state.joint.motor_on & (~state.joint.is_fixed_joint))
        ]
        joint_uniforms = (joint_textures, joint_colours, state.joint.active)
        pixels = joint_renderer(pixels, joint_patch_positions, joint_uniforms)

        # --- Thrusters渲染 (几乎不变, 但我们保存坐标) ---
        thruster_positions_px_float = _world_space_to_pixel_space(state.thruster.global_position) # 保存这个
        thruster_patch_positions = jnp.round(
            thruster_positions_px_float - (float(py_thruster_pixel_size_diagonal) / 2.0)
        ).astype(jnp.int32)
        thruster_textures = coloured_thruster_textures[state.thruster_bindings + 1]
        thruster_rotations = (
            state.thruster.rotation +
            jax.vmap(select_shape, in_axes=(None, 0, None))(state, state.thruster.object_index, static_params).rotation
        )
        thruster_uniforms = (
            jnp.round(thruster_positions_px_float).astype(jnp.int32),
            thruster_rotations, thruster_textures, state.thruster.active
        )
        pixels = thruster_renderer(pixels, thruster_patch_positions, thruster_uniforms)

        # --- 裁剪 (Crop) ---
        py_crop_amount = py_full_screen_size_padding // 2
        final_h, final_w = pixels.shape[0], pixels.shape[1]
        crop_top = min(py_crop_amount, final_h // 2)
        crop_left = min(py_crop_amount, final_w // 2)

        # 裁剪像素坐标
        # 注意：坐标也需要根据裁剪进行平移
        cropped_joint_coords = joint_positions_px_float - jnp.array([crop_left, crop_top])
        cropped_thruster_coords = thruster_positions_px_float - jnp.array([crop_left, crop_top])

        # 裁剪图像
        final_pixels = pixels[crop_top : final_h - crop_top, crop_left : final_w - crop_left]

        # 返回所有需要的数据
        return (
            final_pixels,
            cropped_joint_coords,
            state.joint.motor_on,
            cropped_thruster_coords,
            state.thruster.active
        )

    return render_pixels_and_coords