import numpy as np
import soundfile as sf
import resampy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import spectrogram

def annotated_spectrogram(audio_file, annotations_file, fs_spec=16000,
                            figsize=(10,5), height_ratios=[3, 1]):

    n = np.genfromtxt(annotations_file, dtype=np.str)
    unique_labels = np.unique(n[:,2])

    audio, fs = sf.read(audio_file)

    # extract mono from multichannel audio
    if audio.ndim > 1 and np.min(audio.shape) > 1:
        if np.where(audio.shape == np.min(audio.shape))[0][0] == 0:
            audio = audio[0,:]
        else:
            audio = audio[:,0]

    audio_lowres = resampy.resample(audio, fs, fs_spec)
    f, t, sxx = spectrogram(audio_lowres, fs_spec, nperseg=1024)

    gs = gridspec.GridSpec(2, 1, height_ratios=height_ratios)
    gs.update(hspace=0.1)

    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(gs[0])
    plt.xlim([0,60])
    plt.ylabel('Frequency (Hz)')
    plt.xticks([])
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.pcolormesh(t, f, np.log10(sxx), cmap='GnBu')

    ax2 = plt.subplot(gs[1])
    for i, label in enumerate(unique_labels):
        # get list of start/stop times for each label
        a = n[:,0][n[:,2] == label].astype(float)
        b = n[:,1][n[:,2] == label].astype(float)
        ax2.hlines(np.zeros(len(a)) + i, a, b, colors=plt.cm.tab20(i), lw=10)

    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    plt.xlim([0,60])
    plt.yticks(np.arange(i+1), unique_labels)
    plt.xlabel('Time (seconds)')


def doa_plot(doa_data, plot_labels=False, figsize=(10,10)):

    unique_labels = set(doa_data[:,0])
    colors = [plt.cm.Spectral(each)
                for each in np.linspace(0, 1, len(unique_labels))]

    ax, ax2 = setup_doa_axes(figsize)

    for label in unique_labels:
        source = doa_data[doa_data[:,0] == label]

        if label != -1: # -1 indicates ungrouped points
            # add NaNs to create discontinuities in plot
            pos = np.where(np.abs(np.diff(source[:,2:], axis=0)) >= 360)[0]+1
            theta_nan = np.insert(source[:,2], pos, np.nan)
            phi_nan = np.insert(source[:,3], pos, np.nan)
            t_nan = np.insert(source[:,1], pos, np.nan)

            ax.plot(t_nan, theta_nan,
                    c='black', zorder=10, alpha=1, linewidth=3)
            ax2.plot(t_nan, phi_nan,
                    c='black', zorder=10, alpha=1, linewidth=3)

            if plot_labels:
                ax.text(t_nan[0], theta_nan[0]+20, str(int(label)),
                        backgroundcolor=colors[int(label)])
                ax2.text(t_nan[0], phi_nan[0]+10, str(int(label)),
                         backgroundcolor=colors[int(label)])


def doa_plot_multi(doa_data,
                   zorder=[1,2],
                   plt_colors=[False, True],
                   alpha=[1,1],
                   linestyle=['--','-'],
                   linewidth=[3,3],
                   colors=['black', 'red'],
                   figsize=(10,10)):
    ### pass the ground truth doa data and the predictions in as a list

    ax, ax2 = setup_doa_axes(figsize)

    for i, entry in enumerate(doa_data):

        unique_labels = set(entry[:,0])

        for label in unique_labels:
            source = entry[entry[:,0] == label]

            if label != -1: # -1 indicates ungrouped points
                # add NaNs to create discontinuities in plot
                pos = np.where(np.abs(np.diff(source[:,2:], axis=0))>=360)[0]+1
                theta_nan = np.insert(source[:,2], pos, np.nan)
                phi_nan = np.insert(source[:,3], pos, np.nan)
                t_nan = np.insert(source[:,1], pos, np.nan)

                ax.plot(t_nan, theta_nan,
                        c=colors[i],
                        zorder=zorder[i],
                        alpha=alpha[i],
                        linewidth=linewidth[i],
                        linestyle=linestyle[i])

                ax2.plot(t_nan, phi_nan,
                         c=colors[i],
                         zorder=zorder[i],
                         alpha=alpha[i],
                         linewidth=linewidth[i],
                         linestyle=linestyle[i])


def doa_scatter(xy_t, color='red', s=.1, figsize=(10,10)):

    time_plot_points = xy_t[:,0]
    azi_plot_points = np.rad2deg(xy_t[:,1])
    elev_plot_points = -(np.rad2deg(xy_t[:,2]) - 90)

    # faff about rescaling angles
    azi_plot_points[azi_plot_points > 180] = azi_plot_points[
        azi_plot_points > 180] - 360

    ax, ax2 = setup_doa_axes(figsize)

    ax.scatter(time_plot_points, azi_plot_points, s, color)
    ax2.scatter(time_plot_points, elev_plot_points, s, color)


def setup_doa_axes(figsize):
    gs = gridspec.GridSpec(2, 1, height_ratios=[1,1])
    gs.update(hspace=0.05)

    fig = plt.figure(figsize=figsize)

    ax = plt.subplot(gs[0])
    plt.xticks([])
    plt.yticks(ticks=np.arange(-180,190,40))
    plt.ylim([-190,190])
    plt.ylabel('Azimuth (Degrees)')

    ax2 = plt.subplot(gs[1])
    plt.yticks(ticks=np.arange(-90,100,20))
    plt.ylim([-100,100])
    plt.ylabel('Elevation (Degrees)')
    plt.xlabel('Time (Seconds)')

    return ax, ax2


# ########### plot hammer-projection contours ###########
# # for use in Jupyter notebook, need to set %matplotlib notebook
# N_frames = power_map.shape[1]
#
# fig = plt.figure(figsize=(7,7))
# ax = fig.add_subplot(111, projection='hammer')
#
# for i in range(N_frames):
#     theta_i = xy_t[xy_t[:,0]==i][:,1]
#     phi_i = xy_t[xy_t[:,0]==i][:,2]
#
#     # Plotting the results
#     ax.clear()
#
#     x = theta_look; y = phi_look; z = power_map[:,i]
#     N = int(np.ceil(np.sqrt(6000)))
#     xi, yi = np.meshgrid(np.linspace(0, 2*np.pi, N),
#                            np.linspace(0, np.pi, N))
#     # interpolate fibonacci sample points to grid
#     zi = griddata((x, y), z, (xi, yi))
#
#     ax.contour((xi-np.pi),-(yi-np.pi/2),zi,cmap='Blues', linewidths=.5)
# #     ax.contourf((xi-np.pi),-(yi-np.pi/2),zi,cmap='Blues')#, 10, cmap='Blues')
#     ax.scatter((theta_i-np.pi), -(phi_i-np.pi/2), color='r', marker='x')
#
#     ax.grid(True, alpha=.7, linestyle=':')
#
#     # custom x-axis
#     ax.text(.49, -.15, '0', ha='left', transform=ax.transAxes)
#     ax.text(.24, -.15, r'$\frac{\pi}{2}$', ha='left', transform=ax.transAxes)
#     ax.text(0, -.15, r"$\pi$", ha='left', transform=ax.transAxes)
#     ax.text(.74, -.15, r"$\frac{3\pi}{2}$", ha='left', transform=ax.transAxes)
#     ax.text(.98, -.15, r"$\pi$", ha='left', transform=ax.transAxes)
#
#     ax.annotate('', xy=(.96, -.1275), xycoords='axes fraction', xytext=(.78, -.1275),
#             arrowprops=dict(arrowstyle="<-", alpha=.35))
#     ax.annotate('', xy=(.72, -.1275), xycoords='axes fraction', xytext=(.52, -.1275),
#             arrowprops=dict(arrowstyle="<-", alpha=.35))
#     ax.annotate('', xy=(.48, -.1275), xycoords='axes fraction', xytext=(.27, -.1275),
#             arrowprops=dict(arrowstyle="<-", alpha=.35))
#     ax.annotate('', xy=(.23, -.1275), xycoords='axes fraction', xytext=(.03, -.1275),
#             arrowprops=dict(arrowstyle="<-", alpha=.35))
#
#     # custom y-axis
#     ax.text(.15, .98, '0', ha='left', transform=ax.transAxes)
#     ax.text(.15, -.03, r"$\pi$", ha='left', transform=ax.transAxes)
#
#     ax.annotate('', xy=(.16, .12), xycoords='axes fraction', xytext=(.16, .01),
#             arrowprops=dict(arrowstyle="<-", alpha=.35))
#     ax.annotate('', xy=(.161, .97), xycoords='axes fraction', xytext=(.161, .87),
#             arrowprops=dict(arrowstyle="-", alpha=.35))
#
#     ax.set_xticklabels([])
#     ax.set_yticklabels([])
#
#     ax.set_xlabel(r'Azimuth $\theta$ (radians)', labelpad=35)
#     ax.set_ylabel(r'Inclination $\phi$ (radians)')
#
#     fig.canvas.draw()
