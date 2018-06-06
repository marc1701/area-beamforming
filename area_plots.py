########### plot 3D trajectories w/scatter ###########
# comment out scatter lines to remove scatter
# set up graph axis
fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d', 'aspect':'equal'}, figsize=(10,10))

unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]

for label in unique_labels:

    source = src_out[src_out[:,0] == label]
    doa_pts = xy_t[labels == label]

    if label != -1:

        # add NaNs to create discontinuities in plot
        pos = np.where(np.abs(np.diff(source[:,2:], axis=0)) >= 0.5)[0]+1
        theta_nan = np.insert(source[:,2], pos, np.nan)
        phi_nan = np.insert(source[:,3], pos, np.nan)
        t_nan = np.insert(source[:,1], pos, np.nan)

        ax.plot(theta_nan, phi_nan, t_nan,
                c=colors[int(label)], zorder=10, alpha=1, linewidth=3)

        ax.scatter(doa_pts[:,1], doa_pts[:,2], doa_pts[:,0],
                   c=colors[int(label)], zorder=10, alpha=.1)

    else:
        ax.scatter(doa_pts[:,1], doa_pts[:,2], doa_pts[:,0],
                   c='black', zorder=10, alpha=.1)




########### plot hammer-projection contours ###########
# for use in Jupyter notebook, need to set %matplotlib notebook
N_frames = power_map.shape[1]

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection='hammer')

for i in range(N_frames):
    theta_i = xy_t[xy_t[:,0]==i][:,1]
    phi_i = xy_t[xy_t[:,0]==i][:,2]

    # Plotting the results
    ax.clear()

    x = theta_look; y = phi_look; z = power_map[:,i]
    N = int(np.ceil(np.sqrt(6000)))
    xi, yi = np.meshgrid(np.linspace(0, 2*np.pi, N),
                           np.linspace(0, np.pi, N))
    # interpolate fibonacci sample points to grid
    zi = griddata((x, y), z, (xi, yi))

    ax.contour((xi-np.pi),-(yi-np.pi/2),zi,cmap='Blues', linewidths=.5)
#     ax.contourf((xi-np.pi),-(yi-np.pi/2),zi,cmap='Blues')#, 10, cmap='Blues')
    ax.scatter((theta_i-np.pi), -(phi_i-np.pi/2), color='r', marker='x')

    ax.grid(True, alpha=.7, linestyle=':')

    # custom x-axis
    ax.text(.49, -.15, '0', ha='left', transform=ax.transAxes)
    ax.text(.24, -.15, r'$\frac{\pi}{2}$', ha='left', transform=ax.transAxes)
    ax.text(0, -.15, r"$\pi$", ha='left', transform=ax.transAxes)
    ax.text(.74, -.15, r"$\frac{3\pi}{2}$", ha='left', transform=ax.transAxes)
    ax.text(.98, -.15, r"$\pi$", ha='left', transform=ax.transAxes)

    ax.annotate('', xy=(.96, -.1275), xycoords='axes fraction', xytext=(.78, -.1275),
            arrowprops=dict(arrowstyle="<-", alpha=.35))
    ax.annotate('', xy=(.72, -.1275), xycoords='axes fraction', xytext=(.52, -.1275),
            arrowprops=dict(arrowstyle="<-", alpha=.35))
    ax.annotate('', xy=(.48, -.1275), xycoords='axes fraction', xytext=(.27, -.1275),
            arrowprops=dict(arrowstyle="<-", alpha=.35))
    ax.annotate('', xy=(.23, -.1275), xycoords='axes fraction', xytext=(.03, -.1275),
            arrowprops=dict(arrowstyle="<-", alpha=.35))

    # custom y-axis
    ax.text(.15, .98, '0', ha='left', transform=ax.transAxes)
    ax.text(.15, -.03, r"$\pi$", ha='left', transform=ax.transAxes)

    ax.annotate('', xy=(.16, .12), xycoords='axes fraction', xytext=(.16, .01),
            arrowprops=dict(arrowstyle="<-", alpha=.35))
    ax.annotate('', xy=(.161, .97), xycoords='axes fraction', xytext=(.161, .87),
            arrowprops=dict(arrowstyle="-", alpha=.35))

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax.set_xlabel(r'Azimuth $\theta$ (radians)', labelpad=35)
    ax.set_ylabel(r'Inclination $\phi$ (radians)')

    fig.canvas.draw()
