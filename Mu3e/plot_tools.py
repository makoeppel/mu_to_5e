import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import rc, rcParams
from matplotlib import animation

from . import fastmc as fm

###########################
fsize = 12
fsize_annotate = 10

std_figsize = (1.2 * 3.7, 1.6 * 2.3617)
std_axes_form = [0.16, 0.15, 0.81, 0.76]

electron_color = (0.85, 0.39, 0.14)
positron_color = (0.16, 0.37, 0.65)
# ax_colors = "#CFCFCF"
ax_colors = "darkgrey"
text_color = "grey"

particle_name = {"e+": r"e^+", "e-": r"e^-", "nu": r"\nu"}


# standard figure
def std_fig(ax_form=std_axes_form, figsize=std_figsize, rasterized=False):
    rcparams = {
        "axes.labelsize": fsize,
        "xtick.labelsize": fsize,
        "ytick.labelsize": fsize,
        "figure.figsize": std_figsize,
        "legend.frameon": False,
        "legend.loc": "best",
    }
    plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}\usepackage{amssymb}"
    rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
    rc("text", usetex=True)
    rcParams.update(rcparams)
    mpl.rcParams["hatch.linewidth"] = 0.25
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(ax_form, rasterized=rasterized)
    ax.patch.set_alpha(0.0)

    return fig, ax


# standard saving function
def std_savefig(fig, path, dpi=400, **kwargs):
    dir_tree = os.path.dirname(path)
    os.makedirs(dir_tree, exist_ok=True)
    fig.savefig(path, dpi=dpi, **kwargs)


def get_histogram_and_errors(
    var, decay, bins, color, ax=None, rescale_w=1, mask=None, errorbar=True, **kwargs
):
    if mask is not None:
        values = var[mask]
        weights = decay.weights[mask] * rescale_w
    else:
        values = var
        weights = decay.weights * rescale_w

    pred, bin_edges = np.histogram(values, bins=bins, weights=weights)

    # If performed selection, error on the efficiency needs to be propagated
    if mask is None:
        pred_err2 = np.histogram(values, bins=bins, weights=weights**2)[0]
    else:
        S, _ = np.histogram(var[mask], bins=bins, weights=decay.weights[mask])
        N, _ = np.histogram(var, bins=bins, weights=decay.weights)

        S_pred_err2 = np.histogram(
            var[mask], bins=bins, weights=(decay.weights[mask] * rescale_w) ** 2
        )[0]
        N_pred_err2 = np.histogram(
            var, bins=bins, weights=(decay.weights * rescale_w) ** 2
        )[0]

        eps = S / N
        eps_pred_err2 = N_pred_err2 * (S / N**2) ** 2 + S_pred_err2 * (1 / N**2)
        pred_err2 = eps_pred_err2 * N**2 + N_pred_err2 * (eps) ** 2

    pred_err = np.sqrt(pred_err2)
    bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2
    if ax is not None:
        ax.hist(bin_centers, weights=pred, bins=bin_edges, edgecolor=color, **kwargs)
        #  label=f'kde prediction: {pred.sum():.2g} '\
        # f'$\pm$ {100*np.sqrt(pred_err2.sum())/pred.sum():.2g}%')

        if errorbar:
            for edge_left, edge_right, pred, err in zip(
                bin_edges[:-1], bin_edges[1:], pred, pred_err
            ):
                width = edge_right - edge_left
                if "rwidth" in kwargs.keys():
                    edge_left = (edge_right + edge_left) / 2 - kwargs["rwidth"] / 2
                    edge_right = (edge_right + edge_left) / 2 + kwargs["rwidth"] / 2
                    width = kwargs["rwidth"]

                ax.add_patch(
                    patches.Rectangle(
                        (edge_left, pred - err),
                        width,
                        2 * err,
                        hatch="\\\\\\\\\\\\\\\\",
                        fill=False,
                        linewidth=0,
                        color=color,
                        alpha=0.6,
                        rasterized=True,
                    )
                )

    return pred, pred_err, bin_edges


def EventDraw(
    decay,
    event,
    tot_time=2e-9,
    path=None,
    draw_momentum=False,
    density=False,
    animate=False,
    frames=100,
    plane="xy",
    **kwargs,
):
    if plane == "xy":
        scale_factor = 4.5 / 4
        fig, ax = std_fig(figsize=(std_figsize[0] * scale_factor, std_figsize[1]))
        # if density:
        # draw_decay_density(fig, ax, density, plane=plane)
    elif plane == "zy":
        fig, ax = std_fig(figsize=(10.5, 3.5))
    else:
        raise ValueError(f"Plane {plane} not recognized.")

    # Labels of electron/positron
    ax.plot([], [], label=r"$e^-$", color=electron_color, lw=1, ls="-")
    ax.plot([], [], label=r"$e^+$", color=positron_color, lw=1, ls="-")
    ax.legend(
        frameon=False, fontsize=10, loc=(0.9, 0.85), handlelength=1.2, handletextpad=0.5
    )

    def new_event(time):
        # remove previous event
        for child in ax.get_children():
            if (
                isinstance(child, mpl.text.Annotation)
                or isinstance(child, mpl.lines.Line2D)
                or isinstance(child, mpl.patches.Arc)
            ):
                child.remove()

        i = 0
        for name, p in decay.particles_true.items():
            p = p[event]
            position = decay.pos[event]

            if "e+" in name:
                color = positron_color
            elif "e-" in name:
                color = electron_color
            elif "nu_" in name:
                color = "black"
                if draw_momentum:
                    draw_momentum_track(ax, p, position)
                    i -= 1
                continue
            else:
                continue

            draw_track(
                ax,
                p,
                position,
                name,
                plane=plane,
                tot_time=tot_time,
                fractional_time=time / frames,
                draw_momentum=draw_momentum,
            )
            i += 1

            sub = rf"$p_{{{particle_name[name[:2]]}}}"
            label = (
                rf"{sub} <1$ MeV\\"
                if get_pmag(p) < 1
                else rf"{sub} = {get_pmag(p):.0f}$ MeV"
            )

            ax.annotate(
                label,
                xy=(0.85, i * 0.05),
                fontsize=8,
                color=color,
                xycoords="axes fraction",
                ha="left",
                va="bottom",
                zorder=4,
                alpha=0.75,
            )

        ax.annotate(
            rf"$t = {1e9*tot_time*time / frames:.1f}$~ns",
            xy=(0.01, 0.9),
            fontsize=7,
            color="black",
            xycoords="axes fraction",
            ha="left",
            va="top",
            zorder=4,
            alpha=0.75,
        )

        ax.set_aspect("equal", adjustable="box")

        if plane == "xy":
            XY_geometry(ax)
        elif plane == "zy":
            ZY_geometry(ax, alpha=1, linewidth=1, color="darkgrey")

    new_event(frames)

    if path is None:
        path = f"plots/event_{decay.channel}_{event}.png"

    fig.savefig(path, dpi=400, bbox_inches="tight")

    if animate:
        anim = animation.FuncAnimation(fig, new_event, frames=frames, repeat=False)
        return anim
    else:
        return fig, ax


def draw_decay_density(fig, ax, decay, plane="xy", **kwargs):
    h = ax.hist2d(
        getattr(decay, plane[0]),
        getattr(decay, plane[1]),
        weights=decay.weights / decay.weights.max(),
        bins=(40, 40),
        cmap="Blues",
        cmin=0,
        # cmax=1,
        density=False,
        zorder=0,
        **kwargs,
    )

    cb = fig.colorbar(h[3], fraction=0.025, pad=0.025, location="bottom", aspect=20)
    cb.ax.set_ylabel(
        r"$\mu$ decays", fontsize=8, rotation=0, labelpad=30, color=text_color
    )
    cb.ax.yaxis.set_label_position("right")
    cb.ax.zorder = -1
    cb.ax.tick_params(axis="y", colors=text_color, labelsize=8)
    cb.ax.tick_params(axis="x", colors=text_color, labelsize=8)


def draw_momentum_track(ax, p, pos):
    phi = np.arctan2(p[2], p[1])
    # draw initial direction
    ax.plot(
        [pos[0], pos[0] + 100 * np.cos(phi)],
        [pos[1], pos[1] + 100 * np.sin(phi)],
        color="black",
        lw=0.3,
        ls="--",
    )


def draw_track(
    ax,
    p,
    position,
    name,
    plane="xy",
    fractional_time=1,
    tot_time=1e-9,  # seconds
    draw_momentum=True,
    B=1,  # Tesla
    Npoints=1000,
    **kwargs,
):
    # vel in z direction
    beta_L = p[-1] / p[0]

    # transverse p and velocity
    pT = np.sqrt(p[1] ** 2 + p[2] ** 2)
    beta_T = pT / p[0]
    # theta = np.arccos(p[-1]/get_pmag(p))
    phi = np.arctan2(p[2], p[1])

    arc_R = fm.radius_of_curvature(pT, Bfield=1.0)
    t_exit = fm.time_of_exit(position[-1], beta_L)
    max_arc_angle = (
        t_exit * np.abs(beta_T) * fm.c_light / (2 * np.pi * arc_R) * 2 * np.pi
        if arc_R > 0
        else 0
    )  # degrees

    t_recurl = fm.time_of_recurl(arc_R, beta_T)
    z_recurl = t_recurl * beta_L * fm.c_light

    # Relevant velocity compenents
    v_parallel = beta_L * fm.c_light  # mm / s
    v_perp = beta_T * fm.c_light  # mm / s

    # Period of oscillation
    T = (pT / (0.3)) / (v_perp * B)  # s

    # Time of simulation
    t = np.linspace(
        0,
        tot_time * fractional_time,
        int(Npoints),
    )

    if "e+" in name:
        q = +1
    elif "e-" in name:
        q = -1

    # Helical trajectory
    z = position[-1] + v_parallel * t
    x = position[-3] + (q / 0.28) * (
        p[2] - p[2] * np.cos(q * t / T) + p[1] * np.sin(q * t / T)
    )
    y = position[-2] + (q / 0.28) * (
        -p[1] + p[1] * np.cos(q * t / T) + p[2] * np.sin(q * t / T)
    )

    hit_recurler = (np.abs(z_recurl) > fm.recurler_L / 2 + fm.outer_recurler_gap) & (
        np.abs(z_recurl) < fm.recurler_L / 2 + fm.recurler_L + fm.outer_recurler_gap
    )
    long_track = (np.abs(z_recurl) < fm.recurler_L / 2) | hit_recurler
    short_track = np.max(np.sqrt(x**2 + y**2)) > fm.layer4_R

    # did it pass selection criteria?
    signal_event = short_track  # & long_track
    # if full loop, then just draw circle
    if max_arc_angle > 2 * np.pi:
        max_arc_angle = 2 * np.pi

    arc_kwargs = {
        "lw": (1 if signal_event else 0.5),
        "ls": ((0, (1, 0)) if signal_event else (0, (3, 1))),
        "zorder": 2,
        "alpha": (1 if signal_event else 0.75),
    }
    if "e+" in name:
        arc_kwargs["color"] = positron_color
    elif "e-" in name:
        arc_kwargs["color"] = electron_color
    else:
        raise ValueError(f"Could not draw tracks for: {name}.")

    arc_kwargs.update(kwargs)

    if plane == "xy":
        ax.plot(x, y, **arc_kwargs)
    elif plane == "zy":
        ax.plot(z, y, **arc_kwargs)
    else:
        raise ValueError(f"Could not draw tracks for plane: {plane}.")

    if draw_momentum:
        # draw initial direction
        draw_momentum_track(ax, p, position)


def get_radius_from_apothem(a, vertices):
    return a / np.cos(np.pi / vertices)


def get_pmag(p):
    return np.sqrt(p[3] ** 2 + p[1] ** 2 + p[2] ** 2)


def draw_layer(ax, apothem, vertices, pos=np.array([0, 0]), **kwargs):
    ax.add_patch(
        patches.RegularPolygon(
            pos, vertices, radius=get_radius_from_apothem(apothem, vertices), **kwargs
        )
    )
    kwargs.update({"lw": 0.25})
    ax.add_patch(
        patches.RegularPolygon(
            pos,
            vertices,
            radius=get_radius_from_apothem(apothem, vertices) + 1.5,
            **kwargs,
        )
    )


def ZY_geometry(ax, alpha=0.7, linewidth=1.0, color="gray"):
    # Central detector
    # Inner layer 1
    ax.plot(
        [-fm.layer1_L / 2, fm.layer1_L / 2],
        [fm.layer1_R, fm.layer1_R],
        "-",
        c=color,
        lw=linewidth,
        alpha=alpha,
    )
    ax.plot(
        [-fm.layer1_L / 2, fm.layer1_L / 2],
        [-fm.layer1_R, -fm.layer1_R],
        "-",
        c=color,
        lw=linewidth,
        alpha=alpha,
    )
    ax.plot(
        [-fm.layer1_L / 2, fm.layer1_L / 2],
        [fm.layer1_R + 1.5, fm.layer1_R + 1.5],
        "-",
        c=color,
        lw=linewidth * 0.25,
        alpha=alpha,
    )
    ax.plot(
        [-fm.layer1_L / 2, fm.layer1_L / 2],
        [-fm.layer1_R - 1.5, -fm.layer1_R - 1.5],
        "-",
        c=color,
        lw=linewidth * 0.25,
        alpha=alpha,
    )
    # Inner layer 2
    ax.plot(
        [-fm.layer2_L / 2, fm.layer2_L / 2],
        [fm.layer2_R, fm.layer2_R],
        "-",
        c=color,
        lw=linewidth,
        alpha=alpha,
    )
    ax.plot(
        [-fm.layer2_L / 2, fm.layer2_L / 2],
        [-fm.layer2_R, -fm.layer2_R],
        "-",
        c=color,
        lw=linewidth,
        alpha=alpha,
    )
    ax.plot(
        [-fm.layer2_L / 2, fm.layer2_L / 2],
        [fm.layer2_R + 1.5, fm.layer2_R + 1.5],
        "-",
        c=color,
        lw=linewidth * 0.25,
        alpha=alpha,
    )
    ax.plot(
        [-fm.layer2_L / 2, fm.layer2_L / 2],
        [-fm.layer2_R - 1.5, -fm.layer2_R - 1.5],
        "-",
        c=color,
        lw=linewidth * 0.25,
        alpha=alpha,
    )

    # Outer layer 1
    ax.plot(
        [-fm.layer3_L / 2, fm.layer3_L / 2],
        [fm.layer3_R, fm.layer3_R],
        "-",
        c=color,
        lw=linewidth,
        alpha=alpha,
    )
    ax.plot(
        [-fm.layer3_L / 2, fm.layer3_L / 2],
        [-fm.layer3_R, -fm.layer3_R],
        "-",
        c=color,
        lw=linewidth,
        alpha=alpha,
    )

    # Outer layer 2
    ax.plot(
        [-fm.layer4_L / 2, fm.layer4_L / 2],
        [fm.layer4_R, fm.layer4_R],
        "-",
        c=color,
        lw=linewidth,
        alpha=alpha,
    )
    ax.plot(
        [-fm.layer4_L / 2, fm.layer4_L / 2],
        [-fm.layer4_R, -fm.layer4_R],
        "-",
        c=color,
        lw=linewidth,
        alpha=alpha,
    )

    # Forward recurler
    # Outer layer 1
    ax.plot(
        [
            -fm.recurler_L - fm.layer4_L / 2 - fm.outer_recurler_gap,
            -fm.layer4_L / 2 - fm.outer_recurler_gap,
        ],
        [fm.layer3_R, fm.layer3_R],
        "-",
        c=color,
        lw=linewidth,
        alpha=alpha,
    )
    ax.plot(
        [
            -fm.recurler_L - fm.layer4_L / 2 - fm.outer_recurler_gap,
            -fm.layer4_L / 2 - fm.outer_recurler_gap,
        ],
        [-fm.layer3_R, -fm.layer3_R],
        "-",
        c=color,
        lw=linewidth,
        alpha=alpha,
    )
    # Outer layer 2
    ax.plot(
        [
            -fm.recurler_L - fm.layer4_L / 2 - fm.outer_recurler_gap,
            -fm.layer4_L / 2 - fm.outer_recurler_gap,
        ],
        [fm.layer4_R, fm.layer4_R],
        "-",
        c=color,
        lw=linewidth,
        alpha=alpha,
    )
    ax.plot(
        [
            -fm.recurler_L - fm.layer4_L / 2 - fm.outer_recurler_gap,
            -fm.layer4_L / 2 - fm.outer_recurler_gap,
        ],
        [-fm.layer4_R, -fm.layer4_R],
        "-",
        c=color,
        lw=linewidth,
        alpha=alpha,
    )

    # Backward recurler
    # Outer layer 1
    ax.plot(
        [
            fm.recurler_L + fm.layer4_L / 2 + fm.outer_recurler_gap,
            fm.layer4_L / 2 + fm.outer_recurler_gap,
        ],
        [fm.layer3_R, fm.layer3_R],
        "-",
        c=color,
        lw=linewidth,
        alpha=alpha,
    )
    ax.plot(
        [
            fm.recurler_L + fm.layer4_L / 2 + fm.outer_recurler_gap,
            fm.layer4_L / 2 + fm.outer_recurler_gap,
        ],
        [-fm.layer3_R, -fm.layer3_R],
        "-",
        c=color,
        lw=linewidth,
        alpha=alpha,
    )
    # Outer layer 2
    ax.plot(
        [
            fm.recurler_L + fm.layer4_L / 2 + fm.outer_recurler_gap,
            fm.layer4_L / 2 + fm.outer_recurler_gap,
        ],
        [fm.layer4_R, fm.layer4_R],
        "-",
        c=color,
        lw=linewidth,
        alpha=alpha,
    )
    ax.plot(
        [
            fm.recurler_L + fm.layer4_L / 2 + fm.outer_recurler_gap,
            fm.layer4_L / 2 + fm.outer_recurler_gap,
        ],
        [-fm.layer4_R, -fm.layer4_R],
        "-",
        c=color,
        lw=linewidth,
        alpha=alpha,
    )

    # Styling
    ax.spines[["left", "bottom"]].set_position("center")
    ax.spines[["left", "bottom"]].set_edgecolor(ax_colors)
    ax.spines[["left", "bottom"]].set_zorder(-1)
    ax.spines[["top", "right"]].set_visible(False)

    cap = 500
    xticks = np.linspace(-cap, cap, 5, endpoint=True)
    ax.set_xticks(xticks)
    ax.set_yticks([-150, -100, -50, 50, 100, 150])

    ax.set_xticklabels(
        [
            rf"${xticks[0]:.0f}$",
            rf"${xticks[1]:.0f}$",
            "",
            rf"${xticks[3]:.0f}$",
            rf"${xticks[4]:.0f}$",
        ],
        fontsize=10,
        color=ax_colors,
        zorder=-1,
    )
    ax.set_yticklabels(
        [r"$-150$", "", r"$-50$", r"$50$", "", r"$150$"],
        fontsize=10,
        color=ax_colors,
        zorder=-1,
    )
    ax.tick_params(axis="x", colors=ax_colors, direction="inout", length=3, zorder=-1)
    ax.tick_params(axis="y", colors=ax_colors, direction="inout", length=3, zorder=-1)

    ax.set_xlim(-cap, cap)
    ax.set_ylim(-150, 150)

    ax.scatter(
        0,
        ax.get_ylim()[1],
        marker=10,
        color=ax_colors,
        linewidth=0.0,
        clip_on=False,
        zorder=-1,
    )
    ax.scatter(
        ax.get_xlim()[1],
        0,
        marker=9,
        color=ax_colors,
        linewidth=0.0,
        clip_on=False,
        zorder=-1,
    )

    ax.text(
        1.02,
        0.53,
        r"$z\mathrm{/mm}$",
        fontsize=10,
        color=ax_colors,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        zorder=-1,
    )
    ax.text(
        0.53,
        1.02,
        r"$y\mathrm{/mm}$",
        fontsize=10,
        color=ax_colors,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        zorder=-1,
    )

    target_coordinates = [(0, 19), (50, 0), (0, -19), (-50, 0)]
    ax.add_patch(
        plt.Polygon(
            target_coordinates,
            edgecolor="black",
            lw=0.5,
            zorder=1,
            facecolor="None",
            hatch="..............",
        )
    )

    return ax


def XY_geometry(ax):
    ax.add_patch(
        patches.Circle(
            (0, 0),
            fm.target_R,
            fill=True,
            edgecolor="black",
            lw=0.5,
            zorder=1,
            facecolor="None",
            # hatch="..............",
        )
    )

    det_kwargs = {"lw": 0.75, "color": "grey", "fill": False, "zorder": 1}
    layers_R = [fm.layer1_R, fm.layer2_R, fm.layer3_R, fm.layer4_R]
    layers_vert = [8, 10, 24, 28]

    for R, vert in zip(layers_R, layers_vert):
        draw_layer(ax, R, vert, **det_kwargs)

    ax.spines[["left", "bottom"]].set_position("center")
    ax.spines[["left", "bottom"]].set_edgecolor(ax_colors)
    ax.spines[["left", "bottom"]].set_zorder(-1)
    ax.spines[["top", "right"]].set_visible(False)
    R = 105
    ax.set_ylim(-R, R)
    ax.set_xlim(-R, R)
    ax.set_xticks([-100, -75, -50, -25, 25, 50, 75, 100])
    ax.set_yticks([-100, -75, -50, -25, 25, 50, 75, 100])
    ax.set_xticklabels(
        [-100, "", -50, "", "", 50, "", 100], fontsize=10, color=ax_colors, zorder=-1
    )
    ax.set_yticklabels(
        [-100, "", -50, "", "", 50, "", 100], fontsize=10, color=ax_colors, zorder=-1
    )
    ax.tick_params(axis="x", colors=ax_colors, direction="inout", length=3, zorder=-1)
    ax.tick_params(axis="y", colors=ax_colors, direction="inout", length=3, zorder=-1)

    ax.scatter(
        0,
        ax.get_ylim()[1],
        marker=10,
        color=ax_colors,
        linewidth=0.0,
        clip_on=False,
        zorder=-1,
    )
    ax.scatter(
        ax.get_xlim()[1],
        0,
        marker=9,
        color=ax_colors,
        linewidth=0.0,
        clip_on=False,
        zorder=-1,
    )

    ax.text(
        1.02,
        0.53,
        r"x/mm",
        fontsize=10,
        color=ax_colors,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        zorder=-1,
    )
    ax.text(
        0.53,
        1.02,
        r"y/mm",
        fontsize=10,
        color=ax_colors,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        zorder=-1,
    )

    return ax
