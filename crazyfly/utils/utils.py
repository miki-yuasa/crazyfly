import matplotlib.pyplot as plt


def get_traj_plot(x, y, z, rot1, rot2, rot3, rot4):
    fig = plt.figure(figsize=(12, 6))

    # 3D trajectory plot
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.plot(x, y, z, marker="o", markersize=3, linewidth=2)
    ax1.scatter(x[0], y[0], z[0], marker="o", color="green", s=50, label="Start")
    ax1.text(
        x[0],
        y[0],
        z[0],
        f"({x[0]:.2f}, {y[0]:.2f}, {z[0]:.2f})",
        color="green",
        fontsize=10,
    )
    ax1.scatter(x[-1], y[-1], z[-1], marker="x", color="red", s=50, label="End")
    ax1.text(
        x[-1],
        y[-1],
        z[-1],
        f"({x[-1]:.2f}, {y[-1]:.2f}, {z[-1]:.2f})",
        color="red",
        fontsize=10,
    )
    ax1.set_title("3D Trajectory")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.legend(fontsize=14)

    # time progression of selected states
    ax2 = fig.add_subplot(122)
    t = range(len(x))
    ax2.plot(t, rot1, label="rot1", marker="o", markersize=5, linewidth=2)  # circle
    ax2.plot(t, rot2, label="rot2", marker="d", markersize=5, linewidth=2)  # diamond
    ax2.plot(t, rot3, label="rot3", marker="x", markersize=5, linewidth=2)  # x
    ax2.plot(t, rot4, label="rot4", marker="s", markersize=5, linewidth=2)  # square

    ax2.set_title("Rotor Speed")
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Value (rad/s)")
    ax2.legend(fontsize=14)

    plt.tight_layout()
    plt.close()

    return fig
