from .util import *
from matplotlib.ticker import FormatStrFormatter
from plotly.subplots import make_subplots

class Ploter(object): 
    #%%
    def __init__(self, save=False, cmap="magma", figsize=(5,3)): #" gray_r afmhot_r"
        self.save    = save
        self.cmap    = cmap
        self.figsize = figsize
        self.colores = {"Colombia":["Reds","lightsalmon","red"], "Argentina":["Blues","lightblue","blue"],
                        "Peru":["Greens","darkseagreen","darkgreen"], "Bolivia":["Purples","plum","purple"],
                        "Brazil":["Greys","lightgray","black"], "Costa Rica":["cool","paleturquoise","teal"],
                        "Uruguay":["copper","peachpuff","orange"], "Chile":["Oranges","bisque","chocolate"],
                        "Ecuador":["GnBu","lightsteelblue","steelblue"], "Venezuela":["RdPu","lightpink","mediumvioletred"] }
    #%%
    def PlotAlphaBeta(self, obj, xlim=(-0.05,.2), ylim=(-0.2,0.9), figsize=None, save=None):
        if "synth" in obj.file_name.split("-"):
            plt.close()
            if figsize is None: figsize=(2*self.figsize[0], 2*self.figsize[1])
            if obj.alpha.max()>0.2: xlim=(-0.05,1.1*obj.alpha.max())
            if obj.beta.max()>0.9:  ylim=(-0.2,1.1*obj.beta.max())
            
            fig = plt.figure(figsize=figsize)
            gs  = GridSpec(2, 2, figure=fig)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[1:, 0])
            ax3 = fig.add_subplot(gs[:, 1])

            fig.suptitle("GridSpec")
            fig.tight_layout(pad=3.0)

            viridis = mpl.colormaps["Blues"]
            c       = viridis(np.linspace(0.3, 1, np.size(obj.time_s)))

            ax1.grid()
            ax1.scatter(obj.time_s, obj.alpha, c=c, label="alfa")
            ax1.set_title('Air-Sac Pressure')
            ax1.set_ylabel('α (a.u.)');
            ax1.set_ylim(xlim);

            ax2.scatter(obj.time_s, obj.beta, c=c, label="beta")
            ax2.set_title('Labial Tension')
            ax2.grid()
            ax2.set_xlabel('Time (s)'); ax2.set_ylabel('β (a.u.)')
            ax2.set_ylim(ylim); ax2.sharex(ax1);

            # ------------- Bogdanov–Takens bifurcation ------------------
            ax3.scatter(obj.alpha, obj.beta, c=c, label="Parameters", marker="_")
            ax3.plot(-1/27, 1/3, 'ko')#, label="Cuspid Point"); 
            ax3.axvline(0, color='red', lw=1)#, label="Hopf Bifurcation")
            ax3.plot(obj.mu1_curves[0], obj.beta_bif, '-g', lw=1)#, label="Saddle-Noddle\nBifurcation"); 
            ax3.text(-0.02,0.6, "Hopf",rotation=90, color="r")
            ax3.text(-0.04,0.39,"CP",  rotation=0,  color="k")
            ax3.text(-0.03,0.02,  "SN",rotation=0,  color="g")
            ax3.text(0.1, 0.005, "SN", rotation=0,  color="g")

            ax3.plot(obj.mu1_curves[1], obj.beta_bif, '-g', lw=1)
            ax3.fill_between(obj.mu1_curves[1], obj.beta_bif, 10, where=obj.mu1_curves[1]>0, color='gray', alpha=0.25)
            ax3.set_ylabel('Tension (a.u.)'); ax3.set_xlabel('Pressure (a.u.)')
            ax3.set_title('Parameter Space')
            ax3.set_xlim(xlim); ax3.set_ylim(ylim)
            ax3.legend()
            
            
            fig.suptitle("Motor Gesture Curves\nAudio Sample: {} - {} - {}".format(obj.file_name[:-6], obj.type, obj.no_syllable), fontsize=20)#, family='fantasy')
            plt.show()
            if save is None: save = self.save
            if save: fig.savefig(obj.paths.images / "{}-{}-{}-MotorGesturesParameters.png".format(obj.file_name,obj.id,obj.no_syllable), transparent=True, bbox_inches='tight')
            
            return fig, gs
        else: print("This  is not a synthetic syllable, remember create a synthetic file using the funcion bs.Solve().")
    
    #%%
    def PlotVs(self,obj, xlim=(), figsize=None, save=None):
        if "synth" in obj.file_name.split("-"):
            plt.close()
            if len(xlim)==0: xlim=(obj.timesVs[0], obj.timesVs[-1])
            if figsize is None: figsize=(2*self.figsize[0], 2*self.figsize[1])
            fig, ax = plt.subplots(2, 2, figsize=figsize, sharex=True)

            ax[0,0].plot(obj.timesVs, obj.Vs[:,1], color='g') 
            ax[0,0].set_ylabel("$p_{in}$")
            ax[0,0].set_title(r"Trachea Input Pressure ($p_{in}$)")
            ax[0,0].set_xlim(xlim); 
            
            ax[0,1].plot(obj.timesVs, obj.Vs[:,4], color='b')
            ax[0,1].set_ylabel("$p_{out}$")
            ax[0,1].set_title(r"Trachea Output Pressure ($p_{out}$)")
            ax[0,1].set_xlim(xlim); #ax[0,1].sharex(ax[0,0])
            
            ax[1,0].plot(obj.timesVs, obj.Vs[:,0], color='r')
            ax[1,0].set_xlabel("Time (s)")
            ax[1,0].set_ylabel("$x(t)$")
            ax[1,0].set_title(r"Labial Walls Displacement ($x(t)$)")
            ax[1,0].set_xlim(xlim); #ax[1,0].sharex(ax[0,1]); 
            
            ax[1,1].plot(obj.timesVs, obj.Vs[:,0], color='m')
            ax[1,1].set_xlabel("Time (s)")
            ax[1,1].set_ylabel("$y(t)$");
            ax[1,1].set_title(r"Labial Walls Velocity ($y(t)$)")
            ax[1,1].set_xlim(xlim);# ax[1,1].sharex(ax[1,0]); 

            fig.suptitle('Physical Model Variables\nAudio Sample: {} - {} - {}'.format(obj.file_name[:-6], obj.type, obj.no_syllable), fontsize=20)#\nAudio:'+str(obj.file_name), fontsize=20)
            fig.tight_layout()
            plt.show()
            
            if save is None: save = self.save
            if save: fig.savefig(obj.paths.images / "{}-{}-{}-System-Variables.png".format(obj.file_name,obj.id,obj.no_syllable), transparent=True, bbox_inches='tight')
            return fig, ax
            
        else:  print("This is not a synthetic syllable object, there is not motor gestures asociated to it.")
            
    #%%
    def Plot(self, obj, syllable=None, chunck=None, FF_on=False, SelectTime_on=False, xlim=None, figsize=None, save=None): 
        ticks = ticker.FuncFormatter(lambda x, pos: '{:g}'.format(x*1e-3))
        ticks_x = ticker.FuncFormatter(lambda x, pos: '{:.2f}'.format(x+obj.t0+obj.t0_bs))
        if figsize is None: figsize=(8*self.figsize[0]//5, self.figsize[1]*5/3)
                
        if xlim is not None: xlim = (xlim[0]-obj.t0, xlim[1]-obj.t0)
        else:                xlim = (obj.time[0], obj.time[-1])

        if syllable!=None: syllable_on, ratios = 1, [1,2,1]
        else:              syllable_on, ratios = 0, [1,2]

        plt.close()
        
        if "birdsong" in obj.id:            
            fig, ax = plt.subplots(2+int(syllable_on), 1, gridspec_kw={'height_ratios': ratios}, figsize=figsize, sharex=True)
            
            img = Specshow(obj.Sxx_dB, x_axis="s", y_axis="linear", sr=obj.fs,
                           hop_length=obj.hop_length, ax=ax[1], cmap=self.cmap)
            
            if FF_on: 
                if obj.ff_method=="yin" or obj.ff_method=="pyin" or obj.ff_method=="manual":
                    ax[1].plot(obj.time, obj.FF,  "bo", label=r"FF$_{{}}$".format(obj.ff_method), ms=8)
                elif obj.ff_method=="both":
                    ax[1].plot(obj.time, obj.FF,  "co", label=r"FF$_{pyin}$", ms=8)
                    ax[1].plot(obj.time, obj.FF2, "b*", label=r"FF$_{yin}$", ms=8)
                #ax[1].legend()
                ax[1].legend(bbox_to_anchor=(1.01, 0.65))
            # for i in range(len(obj.syllables)):  
            #     ax[0].plot([obj.time[obj.syllables[i][0]], obj.time[obj.syllables[i][-1]]], [0, 0], 'white', lw=5)
            #     ax[0].text((obj.time[obj.syllables[i][-1]]-obj.time[obj.syllables[i][0]])/2, 0.5, str(i))
            ax[1].set_ylim(obj.flim)
            ax[1].set_xlim(xlim)
            ax[1].yaxis.set_major_formatter(ticks)
            ax[1].xaxis.set_major_formatter(ticks_x)
            ax[1].set_ylabel('Frequency (kHz)'); 
            ax[1].set_xlabel('Time (s)');
            

            ax[0].plot(obj.time_s, obj.s,'k', label='waveform')
            ax[0].plot(obj.time_s, np.ones(obj.time_s.size)*obj.umbral, '--', label='umbral')
            ax[0].plot(obj.time_s, obj.envelope, label='envelope')
            ax[0].legend(bbox_to_anchor=(1.01, 0.45))
            ax[0].xaxis.set_major_formatter(ticks_x)
            ax[0].set_ylabel('Amplitude (a.u)'); ax[0].set_xlabel(''); 
            #ax[1].sharex(ax[0])

            if chunck!=None:
                ax[1].plot(chunck.time+chunck.t0, chunck.FF, 'gv', label='Chunck', ms=10)
                ax[2].plot(chunck.time+chunck.t0-syllable.t0, chunck.FF, 'gv', label='Chunck', ms=8)

            if syllable!=None:
                ax[1].plot(syllable.time+syllable.t0, syllable.FF, 'b+', label='Syllable'.format(syllable.fs), ms=6)
                

                img = Specshow(syllable.Sxx_dB, x_axis="s", y_axis="linear", sr=syllable.fs,
                               hop_length=syllable.hop_length, ax=ax[2], cmap=self.cmap)
                
                ax[2].plot(syllable.time, syllable.FF, 'b+', label='Syllable', ms=15)
                ax[2].set_ylim(obj.flim); 
                # if xlim is not None:
                #     x_list = np.linspace(xlim[0], xlim[1], 10)
                #     ax[2].set_xlim(xlim); ax[2].set_xticks(x_list, round(x_list,2))
                # else:
                #     ax[2].set_xlim((syllable.time[0], syllable.time[-1]))
            
                ax[2].legend(loc='upper right', title="FF")
                ax[2].set_xlabel('Time (s)'); ax[2].set_ylabel('f (khz)')
                ax[2].set_title('Single Syllable Spectrum, No {}'.format(syllable.no_syllable))
                ax[2].yaxis.set_major_formatter(ticks)
                ax[2].xaxis.set_major_formatter(ticks_x)
                
                ax[1].legend(loc='upper right', title="FF")
                path_save = obj.paths.images / "{}-{}-AllSongAndSyllable.png".format(obj.file_name[:-4], syllable.no_syllable)
            else: path_save = obj.paths.images / "{}-AllSongAndSyllable.png".format(obj.file_name[:-4])

            fig.suptitle("Audio Sample: "+obj.file_name, fontsize=18)
            if SelectTime_on==True:  self.klicker = Klicker(fig, ax[1])
            fig.tight_layout()
            plt.show()
        
            if save is None: save = self.save
            if save: fig.savefig(path_save, transparent=True, bbox_inches='tight')
        else:  # syllable ------------------------------------------------
            fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]}, sharex=True,
                                   figsize=figsize)

            img = Specshow(obj.Sxx_dB, x_axis="s", y_axis="linear", sr=obj.fs,
                           hop_length=obj.hop_length, ax=ax[0])
            ax[0].yaxis.set_major_formatter(ticks)
            ax[0].xaxis.set_major_formatter(ticks_x)
            
            if   SelectTime_on==True and FF_on==False:
                self.klicker = Klicker(fig, ax[0])
            elif SelectTime_on==True and FF_on==True:
                if obj.ff_method=="yin" or obj.ff_method=="pyin" or obj.ff_method=="manual":
                    ax[0].plot(obj.time, obj.FF,"co", ms=7, label=r"FF$_{{}}$".format(obj.ff_method))
                elif obj.ff_method=="both":
                    ax[0].plot(obj.time, obj.FF,"co", ms=7)
                    ax[0].plot(obj.time, obj.FF2,"b*", ms=7)
                self.klicker = Klicker(fig, ax[0])
            elif SelectTime_on==False and FF_on==True:
                if obj.ff_method=="yin" or obj.ff_method=="pyin" or obj.ff_method=="manual":
                    ax[0].plot(obj.time, obj.FF,"co", ms=7, label=r"FF$_{{}}$".format(obj.ff_method))
                elif obj.ff_method=="both":
                    ax[0].plot(obj.time, obj.FF,"co", ms=7,label=r"FF$_{pyin}")
                    ax[0].plot(obj.time, obj.FF2,"b*", ms=7,label=r"FF$_{yi}$")
                ax[0].legend()
            
            ax[0].set_ylim(obj.flim); #ax[0].set_xlim(min(obj.time), max(obj.time));
            ax[0].set_xlim(xlim)
            ax[0].set_ylabel('Frequency (kHz)'); ax[0].set_xlabel(''); 

            ax[1].plot(obj.time_s, obj.s,'k', label='waveform')
            ax[1].plot(obj.time_s, obj.envelope, label='envelope')
            ax[1].legend(bbox_to_anchor=(1.01, 0.65))
            ax[1].xaxis.set_major_formatter(ticks_x)
            ax[1].set_xlabel('Time (s)'); ax[1].set_ylabel('Amplitude (a.u)');
            #ax[1].sharex(ax[0])
            
            # if xlim is not None:
            #     x_list = np.linspace(xlim[0], xlim[1], 10)
            #     ax[0].set_xlim(xlim); ax[0].set_xticks(x_list, round(x_list,2))
            # else:
            #     ax[0].set_xlim((obj.time[0], obj.time[-1]))
            
            
            fig.suptitle("Audio Sample: {}\n{} - {}".format(obj.file_name, obj.type, obj.no_syllable), fontsize=18) # 'Audio:\n{}'.format(obj.file_name[:-4])
            fig.tight_layout()
            plt.show()

            path_save = obj.paths.images / "{}-{}-{}.png".format(obj.file_name[:-4], obj.type, obj.no_syllable)
            
            if save is None: save = self.save
            if save: fig.savefig(path_save, transparent=True, bbox_inches='tight')
            return fig, ax

    #%%    
    def Syllables(self, obj, obj_synth, FF_on=False, figsize=None, save=None): 
        plt.close()
        ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x*1e-3))
        if figsize is None: figsize=(2*self.figsize[0]+2, 2*self.figsize[1])
        fig, ax = plt.subplots(2, 2, figsize=figsize, sharex=True)
        fig.subplots_adjust(right=0.9)

        ax[1,0].plot(obj.time_s, obj.s, label='waveform', c='b')
        ax[1,0].set_xlim((obj.time_s[0], obj.time_s[-1]))
        ax[1,0].plot(obj.time_s, obj.envelope, label='envelope', c='darkblue')
        ax[1,0].legend()
        ax[1,0].set_ylabel("Amplitud (a.u.)"); ax[1,0].set_xlabel(" "*80+"Time (s)")
        
        ax[1,1].plot(obj_synth.time_s, obj_synth.s, label='waveform', c='g')
        ax[1,1].plot(obj_synth.time_s, obj_synth.envelope, label='envelope', c='darkgreen')
        ax[1,1].legend(); ax[1,1].set_xlabel(''); ax[1,1].set_ylabel("")

        img = Specshow(obj.Sxx_dB, x_axis="s", y_axis="linear", sr=obj.fs,
                       hop_length=obj.hop_length, ax=ax[0,0], cmap=self.cmap)
        ax[0,0].set_title('Real',fontweight='bold') 
        ax[0,0].yaxis.set_major_formatter(ticks);  ax[0,0].set_ylim(obj.flim)
        ax[0,0].set_ylabel('Frequency (kHz)'); ax[0,0].set_xlabel('')
        
        img = Specshow(obj_synth.Sxx_dB, x_axis="s", y_axis="linear", sr=obj_synth.fs,
                       hop_length=obj_synth.hop_length, ax=ax[0,1], cmap=self.cmap)
        cbar_ax = fig.add_axes([0.925, 0.53, 0.015, 0.35])
        clb = fig.colorbar(img, cax=cbar_ax, format="%+2.f")
        clb.set_label('Power (dB)', labelpad=-20, y=1.15, rotation=0)
        ax[0,1].yaxis.set_major_formatter(ticks)
        ax[0,1].set_title('Synthetic',fontweight='bold') 
        ax[0,1].set_ylim(obj.flim)
        ax[0,1].set_ylabel(''); ax[0,1].set_xlabel('')

        if FF_on: 
            ax[0,0].plot(obj.time, obj.FF, 'bo-', lw=2, label="Real FF")
            ax[0,1].plot(obj_synth.time, obj_synth.FF, 'go-', lw=2,  label="Synth FF")
            
            ax[0,0].legend(); ax[0,1].legend();
        
        fig.suptitle('Audio Sample: {} - {} - {}'.format(obj.file_name, obj.type, obj.no_syllable), fontsize=20, y=1.03)
        plt.show();
        
        if save is None: save = self.save
        if save: fig.savefig(obj.paths.images / "{}-{}-{}-SoundAndSpectros.png".format(obj.file_name[:-4],obj.id,obj.no_syllable), transparent=True, bbox_inches='tight')
        
        return fig, ax
    #%%
    def Scores(self, obj, obj_synth, cmp="afmhot_r", figsize=None, ylim=None, save=None):
        plt.close()
        ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x*1e-3))
        ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:.2g}'.format(x))
        if figsize is None: figsize=(2*self.figsize[0], 3*self.figsize[1])
        fig = plt.figure(constrained_layout=False, figsize=figsize)
        gs  = fig.add_gridspec(nrows=9, ncols=5, right=0.7, hspace=0.7)
        #vmin, vmax = obj.Sxx_dB.min(), obj.Sxx_dB.max()
        
        # --------------- scores: FF and SCI ---------------------------------
        ax2 = fig.add_subplot(gs[:2,:])
        ax2.plot(obj_synth.time, 100*obj_synth.deltaFF, "*-", color="k", ms=5, lw=1, alpha=0.8,
                 label=r'FF, $\overline{FF}$='+str(round(100*obj_synth.deltaFF_mean,2)))
        ax2.plot(obj_synth.time, 100*obj_synth.deltaSCI, "*-", color="purple", ms=5, lw=1, alpha=0.8,
                 label=r'SCI, $\overline{SCI}$='+str(round(100*obj_synth.deltaSCI_mean,2)));
        
        ax2.set_xlabel(''); ax2.set_ylabel('Relative Error (%)');
        ax2.legend(bbox_to_anchor=(1.15, 0.55), borderpad=0.6, labelspacing=0.7,)
        ax2.set_xlim((obj_synth.time[0],obj_synth.time[-1]))
        ax2.xaxis.set_major_formatter(ticks_x)
        if ylim is not None: ax2.set_ylim(ylim)
        
        # ------------------ spectrum ---------------
        ax1 = fig.add_subplot(gs[2:5,:], sharex=ax2)
        
        img = Specshow(obj.Sxx_dB, x_axis="s", y_axis="linear", sr=obj.fs,
                       hop_length=obj.hop_length, ax=ax1, cmap=self.cmap)
        ax1.plot(obj.time,        obj.FF,       'b*-', label=r'real',ms=7)
        ax1.plot(obj_synth.time,  obj_synth.FF, 'go-', label=r'synth', ms=3)

        ax1.legend(borderpad=0.6, labelspacing=0.7)
        ax1.set_ylim(obj.flim); ax1.set_xlim((obj.time[0], obj.time[-1]))
        ax1.set_ylabel('Frequency (kHz)'); ax1.set_xlabel('');
        ax1.yaxis.set_major_formatter(ticks)
        ax1.xaxis.set_major_formatter(ticks_x)
        
        # ------------------ SCI -------------------------
        ax12 = fig.add_subplot(gs[5:7,:], sharex=ax2)
        ax12.set_ylabel(r"Similarity (dl)"); 
        
        # --------------- acousitcal features ----------------------
        ax11 = ax12.twinx()
        lr = ax11.plot(obj.time, obj.SCI, 'b*-', label=r'$SCI_{real}$, $\overline{SCI}$='+str(round(obj.SCI.mean(),2)), ms=7)
        ls = ax11.plot(obj.time, obj_synth.SCI, 'go-', label=r'$SCI_{synth}$, $\overline{SCI}$='+str(round(obj_synth.SCI.mean(),2)), ms=5, alpha=0.8)

        lh   = ax12.plot(obj.time, obj_synth.Df, 'H', label=r'DF, $\overline{DF}$='+str(round(obj_synth.Df.mean(),2)), ms=3)
        lskl = ax12.plot(obj.time, obj_synth.SKL, 's', color="purple", label=r'SKL, $\overline{SKL}$='+str(round(obj_synth.SKL.mean(),2)), ms=3)
        lc   = ax12.plot(obj.time, obj_synth.correlation, 'p', label=r'cor, $\overline{corr}$='+str(round(obj_synth.correlation.mean(),2)), ms=3)
        
        ax11.set_ylabel("SCI (dl)"); 
        ax11.set_ylim((0,5)); ax11.legend(bbox_to_anchor=(1.28, 1))
        
        lns = lr+ls+lh+lskl+lc
        labs = [l.get_label() for l in lns]
        ax11.legend(lns, labs, bbox_to_anchor=(1.075, 1.15), title="Acoustical Features", title_fontproperties={'weight':'bold'})

        # ------------------ sound -------------------------
        ax9 = fig.add_subplot(gs[7:,:], sharex=ax2)
        ax9.plot(obj.time_s,       obj.s,              c='blue',  label=r'waveform$_{real}$', alpha=0.8)
        ax9.plot(obj_synth.time_s, obj_synth.s,        c='green', label=r'waveform$_{synth}$', alpha=0.8)
        ax9.plot(obj.time_s,       obj.envelope,       c='darkblue',  label=r'envelope$_{real}$')
        ax9.plot(obj_synth.time_s, obj_synth.envelope, c='darkgreen', label=r'envelope$_{synth}$')
        ax9.set_ylim((0,1))
        ax9.set_xlabel("Time (s)")
        ax9.legend(bbox_to_anchor=(1.15, 0.5)); ax9.set_ylabel("Amplitude (a.u.)")
        
        cbar_ax = fig.add_axes([0.725, 0.47, 0.02, 0.23])
        clb = fig.colorbar(img, cax=cbar_ax)
        clb.set_label('Power (dB)', labelpad=-20, y=1.125, rotation=0)
        
        fig.suptitle("Results of the Scoring Variables\nAudio Sample: {} - {} - {}".format(obj.file_name, obj.type, obj.no_syllable), fontsize=20, y=0.99)
        plt.show()
        
        if save is None: save = self.save
        if save: fig.savefig(obj.paths.images / "{}-{}-{}-Scoring-Variables.png".format(obj.file_name[:-4],obj.id,obj.no_syllable), transparent=True, bbox_inches='tight') 
        return fig, gs

    #%%
    def ComparingSpectros(self, obj, obj_synth, cmp="afmhot_r", figsize=None, ylim=None, save=None, sharey=True):
        plt.close()
        ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x*1e-3))
        if figsize is None: figsize=(3*self.figsize[0], 2*self.figsize[1])
        fig = plt.figure(figsize=figsize)

        gs  = fig.add_gridspec(nrows=2, ncols=3, hspace=0.35)
        vmin, vmax = obj.Sxx_dB.min(), obj.Sxx_dB.max()
        
        # ------------------ spectrogams ----------------------------
        ax3 = fig.add_subplot(gs[0, 0])
        
        img = Specshow(obj.Sxx_dB, x_axis="s", y_axis="linear", sr=obj.fs,
                       hop_length=obj.hop_length, ax=ax3, cmap=self.cmap)
        clb = fig.colorbar(img, ax=ax3)
        clb.set_label('Power (dB)', labelpad=-16, y=-0.095, rotation=0)
        ax3.set_ylim(obj.flim); ax3.set_xlim((obj.time[0], obj.time[-1]))
        ax3.yaxis.set_major_formatter(ticks)
        ax3.tick_params(axis="x", which='both', labelrotation=90)
        ax3.set_ylabel('');  ax3.set_xlabel('');
        ax3.set_title('Real', fontweight="bold")

        if sharey: ax4 = fig.add_subplot(gs[0, 1], sharex=ax3, sharey=ax3)
        else:      ax4 = fig.add_subplot(gs[0, 1], sharex=ax3)
        img = Specshow(obj_synth.Sxx_dB, x_axis="s", y_axis="linear", sr=obj_synth.fs,
                       hop_length=obj_synth.hop_length, ax=ax4, cmap=self.cmap)
        clb = fig.colorbar(img, ax=ax4)
        clb.set_label('Power (dB)', labelpad=-16, y=-0.095, rotation=0)
        ax4.set_ylim(obj.flim)
        ax4.yaxis.set_major_formatter(ticks)
        ax4.tick_params(axis="x", which='both', labelrotation=90)
        ax4.set_title("Synthetic", fontweight="bold")
        ax4.set_ylabel(''); ax4.set_xlabel(''); 
        
        # ------------------ Mel spectgrograms ------------------
        if sharey: ax5 = fig.add_subplot(gs[1, 0], sharex=ax3, sharey=ax3)
        else:      ax5 = fig.add_subplot(gs[1, 0], sharex=ax3)
        img = Specshow(obj.FF_coef, x_axis="s", y_axis="linear", sr=obj.fs,
                       hop_length=obj.hop_length, ax=ax5, cmap=self.cmap, vmin=0, vmax=100)
        clb = fig.colorbar(img, ax=ax5)
        ax5.set_ylim(obj.flim); 
        ax5.yaxis.set_major_formatter(ticks)
        ax5.tick_params(axis="x", which='both', labelrotation=90)
        ax5.set_ylabel(''); ax5.set_xlabel('');
        ax5.set_ylabel(" "*68+'Frequency (kHz)'); ax5.set_xlabel(''); 
        
        if sharey: ax6 = fig.add_subplot(gs[1, 1], sharex=ax3, sharey=ax3)
        else:      ax6 = fig.add_subplot(gs[1, 1], sharex=ax3)
        img = Specshow(obj_synth.FF_coef, x_axis="s", y_axis="linear", sr=obj.fs,
                       hop_length=obj_synth.hop_length, ax=ax6, cmap=self.cmap, vmin=0, vmax=100)
        clb = fig.colorbar(img, ax=ax6)
        ax6.set_ylim(obj.flim); 
        ax6.yaxis.set_major_formatter(ticks)
        ax6.tick_params(axis="x", which='both', labelrotation=90)
        ax6.set_ylabel(''); ax6.set_xlabel('Time (s)');     
        
        # ------------------ Delta Sxx - Mel ------------------------
        if sharey: ax7 = fig.add_subplot(gs[0, 2], sharex=ax3, sharey=ax3)
        else:      ax7 = fig.add_subplot(gs[0, 2], sharex=ax3)
        img = Specshow(obj_synth.deltaSxx, x_axis="s", y_axis="linear", sr=obj.fs, vmin=0, vmax=1,
                       hop_length=obj_synth.hop_length, ax=ax7, cmap=self.cmap)
        clb = fig.colorbar(img, ax=ax7)
        clb.set_label('Power (dB)', labelpad=-14, y=-0.095, rotation=0)
        ax7.yaxis.set_major_formatter(ticks)
        ax7.tick_params(axis="x", which='both', labelrotation=90)
        ax7.set_ylabel(''); ax7.set_xlabel('');
        ax7.set_ylim(obj.flim);  
        ax7.set_title(r'Difference ($\Delta$)', fontweight="bold")
        #ax7.text(0.075, -9.9,"Regular", fontweight="bold", rotation=90, fontsize=12)
        
        if sharey: ax8 = fig.add_subplot(gs[1, 2], sharex=ax3, sharey=ax3)
        else:      ax8 = fig.add_subplot(gs[1, 2], sharex=ax3)
        
        img = Specshow(obj_synth.deltaMel, x_axis="s", y_axis="linear", sr=obj.fs,
                       hop_length=obj_synth.hop_length, ax=ax8, cmap=self.cmap)
        ax8.set_ylabel(''); ax8.set_xlabel('');  
        ax8.set_ylim(obj.flim);
        ax8.yaxis.set_major_formatter(ticks)
        
        fig.colorbar(img, ax=ax8)
        #ax8.text(0.075, 10.2,"Mel", fontweight="bold", rotation=90, fontsize=12)
        ax8.tick_params(axis="x", which='both', labelrotation=90)
        
            
        fig.suptitle("Result of Comparing Spectral Content\nAudio Sample: {} - {} - {}".format(obj.file_name, obj.type, obj.no_syllable), fontsize=20, y=1.05)
        plt.show()
        
        if save is None: save = self.save
        if save: fig.savefig(obj.paths.images / "{}-{}-{}-Comparing-Spectros.png".format(obj.file_name[:-4],obj.id,obj.no_syllable), transparent=True, bbox_inches='tight') 
        return fig, gs

    #%%    
    def FindTimes(self, obj, FF_on=False):
        plt.close()
        ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x*1e-3))
        fig, ax = plt.subplots(1, 1, figsize=(self.figsize[0]+1, self.figsize[1]+2))#, constrained_layout=True)
        
        img = Specshow(obj.Sxx_dB, x_axis="s", y_axis="linear", sr=obj.fs,
                       hop_length=obj.hop_length, ax=ax, cmap=self.cmap)

        if FF_on:
            ax.plot(obj.time, obj.FF, "co", ms=8)#, label="Fundamental Frequency")
        
        for i in range(len(obj.syllables)):  
            ax[0].plot([obj.time[obj.syllables[i][0]], obj.time[obj.syllables[i][-1]]], [0, 0], color='white', lw=5)
            ax[0].text((obj.time[obj.syllables[i][-1]]-obj.time[obj.syllables[i][0]])/2, 0.5, str(i), color='white')
        ax.set_ylim(obj.flim); ax.set_xlim(min(obj.time), max(obj.time));
        ax.set_title("Song Spectrum"); 
        ax.yaxis.set_major_formatter(ticks)
        ax.set_ylabel('f (kHz)');
        
        klicker = Klicker(fig, ax)
        plt.show()
        
        return klicker
    
    #%%
    def Plot3d(self, obj):
        plt.close()
        
        fig = plt.figure(constrained_layout=True)
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(obj.times, obj.freqs, obj.Sxx_dB, cmap=self.cmap)
        ax.set_xlabel("time (s)"); ax.set_ylabel("Frequency (Hz)")
        ax.set_zlabel("Power (dB)")
        
        plt.show()
        
        
    #%%
    def PlotCountries(self, all_rates, method="matplotlib", alpha=0.5, xlim=(-90,-30), size=[800,500]):
        max_altitude = 4300
        col = all_rates["Country"].unique()
        dark = [self.colores[k][2] for k in col]
        light = [self.colores[k][1] for k in col]
        
        plt.close()

        if method=="matplotlib":
            #figsize = (9,8)

            gdf = geopandas.GeoDataFrame(all_rates, geometry=geopandas.points_from_xy(all_rates.Longitude, all_rates.Latitude))
            world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
            ax = world[world.continent == 'South America'].plot(color='white', edgecolor='black', alpha=0.5)
            
            for k in ["Panama","Nicaragua","Honduras","El Salvador","Guatemala"]:
                world[world.name == k].plot(color='white', edgecolor='black', alpha=0.5, ax=ax)
                
            for k in self.colores.keys():
                world[world.name == k].plot(edgecolor=self.colores[k][2], color=self.colores[k][1], ax=ax, alpha=alpha)
            
            plotted = []
            for i in range(len(all_rates)):
                viridis = mpl.colormaps[self.colores[all_rates.iloc[i]["Country"]][0]]
                c = viridis(np.linspace(0.5, 1, 10))
                color = c[int(10*all_rates.iloc[i]["Altitude"]/max_altitude)]
                
                if all_rates.iloc[i]["Country"] not in plotted:
                        label = all_rates.iloc[i]["Country"]
                        plotted.append(label)
                else:   label = ""
                ax.scatter(all_rates.iloc[i]["Longitude"], all_rates.iloc[i]["Latitude"], color=color, marker="x", label=label)

            ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
            ax.legend(title="Country", bbox_to_anchor=(1.1, 0.65), title_fontproperties={'weight':'bold'})
            ax.set_xlim(xlim); ax.set_ylim((-60,20))
        
        elif method=="plotly":
            app = Dash(__name__)
            app.layout = html.Div([
                #html.H4('Interactive scatter plot with Iris dataset'),
                dcc.Graph(id="scatter-plot"),
                html.P("Filter by Altitude (m):"),
                dcc.RangeSlider(id='range-slider',
                                min=0, max=4200, step=11,
                                marks={i: '{}'.format(i) for i in range(0,4400,200)},
                                value=[0,4200]), ])

            @app.callback(
                Output("scatter-plot", "figure"), 
                Input("range-slider", "value"))

            def update_bar_chart(slider_range):
                low, high = slider_range
                
                mask = (all_rates['Altitude'] > low) & (all_rates['Altitude'] < high)
                col = all_rates[mask]["Country"].unique()
                dark = [self.colores[k][2] for k in col]
                light = [self.colores[k][1] for k in col]

                fig = px.choropleth(locations=col, locationmode="country names", scope="world", color=col,
                                    color_discrete_sequence=light, width=size[0], height=size[1])
                fig.update_traces(showlegend=False)
                fig.add_traces(list(px.scatter_geo(all_rates[mask], lon="Longitude", lat="Latitude", symbol_sequence="x",
                                                    hover_name="Filename", hover_data=["Longitude", "Latitude", "Altitude"],
                                                    color="Country", color_discrete_sequence=dark ).select_traces() ))
                fig.update_layout(legend_title_text='<b>      Country<b>',
                                legend=dict(bordercolor="gray", borderwidth=1,
                                            yanchor="top", y=0.99, xanchor="left", x=0.99)
                                )
                fig.update_geos(center=dict(lon=-60, lat=-14), landcolor="whitesmoke", showcountries=True,
                                lataxis_range=[-50,40], lonaxis_range=[-40, 40] )

                return fig
            app.run_server(debug=True)
            plt.show()
        
    def PlotRates(self, rates, method="matplotlib",  figsize=(9,5), marker=[".",5], xlim=(0,70), ylim=(0,10),
                  show_label=False, error=True, lines=False, maxlines=False, points=True, maxponits=False,
                  shapes=False, color_div=False):
        size = (figsize[0]*100, figsize[1]*100)
        
        col = rates["Country"].unique()
        dark = [self.colores[k][2] for k in col]
        light = [self.colores[k][1] for k in col]
        # PODOS line
        x1, y1 = 5.313432835820896, 6.827004219409282
        x2, y2 = 39.343283582089555, 2.4219409282700424
        m = (y2-y1)/(x2-x1)
        b = y1-m*x1
        m, b = -0.124, 7.55 #3 PODOS 2017
        def y_podos(x): return m*x+b
        x = np.linspace(0,120)
        max_altitude = 4300
        def MaxPoints(all_rates):
            #trs, bws, c = [], [], 5
            c = 5
            indxs = pd.DataFrame(columns=all_rates.columns)
            for i in range(20):
                low, high = c*i, c*(i+1)
                mask = (all_rates["Trill Rate"]>=low) & (all_rates["Trill Rate"]<high)
                if len(all_rates[mask])>0:
                    index = all_rates[mask]["Band Width"].argmax()
                    tr, bw = all_rates[mask].iloc[index][["Trill Rate", "Band Width"]]
                    indxs = pd.concat([indxs, all_rates[mask].iloc[index].to_frame().T], axis=0, ignore_index=True)
                    #trs.append(tr); bws.append(bw)
            return indxs#trs, bws
        
        if method=="matplotlib":
            plotted, country = [], []
            alphabet,calp = string.ascii_uppercase,0
            text_x, text_y = 0.5, 0.1
            
            plt.close()
            fig = plt.figure(figsize=figsize, constrained_layout=True)
            plt.plot(x, y_podos(x), "--", color="dimgray", alpha=0.6, linewidth=1)
            
            for i in range(len(rates)):    
                if rates.iloc[i]["Country"] in country: calp+=1
                else:                                   calp=0; country.append(rates.iloc[i]["Country"])
                
                viridis = mpl.colormaps[self.colores[rates.iloc[i]["Country"]][0]]
                c = viridis(np.linspace(0.25, 1, 10))
                if color_div:
                    color = c[int(10*rates.iloc[i]["Altitude"]/max_altitude)]
                elif not color_div:
                    color = self.colores[rates.iloc[i]["Country"]][2]
                
                if show_label:
                    label = str(calp)+"-"+rates.iloc[i]["Country-State-County"]+":"+str(rates.iloc[i]["file_name"])
                    plt.text(rates.iloc[i]["Trill Rate"]+text_x, rates.iloc[i]["f_mean"]*1e-3+text_y, str(calp), color=color)
                    b_box, title, ncol = (0.75, -0.20), "Label-Country-State:File_name", 3
                else:
                    if str(rates.iloc[i]["Filename"]) not in plotted and calp==0:
                            label = rates.iloc[i]["Country"]
                            plotted.append(label)
                            b_box, title, ncol = (1.2, 0.9), "Country", 1
                    else:   label=""
                if error:
                    plt.errorbar(rates.iloc[i]["Trill Rate"], rates.iloc[i]["Band Width"]*1e-3, label=label,
                                xerr=rates.iloc[i]["tr_e"], yerr=rates.iloc[i]["BW_e"]*1e-3, 
                                fmt=marker[0], markersize=marker[1], color=color, alpha=0.8)
                elif not error:
                    plt.scatter(rates.iloc[i]["Trill Rate"], rates.iloc[i]["Band Width"]*1e-3, label=label,
                                marker=marker[0], s=marker[1], color=color, alpha=0.8)
                
            plt.xlim(xlim); plt.ylim(ylim);
            plt.xlabel("Trill Rate (Hz)"); plt.ylabel("Band Width (kHz)");
            plt.legend(title=title, bbox_to_anchor=b_box, title_fontproperties={'weight':'bold'}, ncol=ncol)

            return fig
        
        elif method=="plotly":
            plt.close()
            all_rates = rates.copy()
            all_rates["Band Width"] *=1e-3
            all_rates["BW_e"] *=1e-3

            app = Dash(__name__)
            app.layout = html.Div([
                #html.H4('Interactive scatter plot with Iris dataset'),
                dcc.Graph(id="scatter-plot"),
                html.P("Filter by Altitude (m):"),
                dcc.RangeSlider(id='range-slider', min=-1, max=4200, step=11, value=[0,4200],
                                marks={i: '{}'.format(i) for i in range(0,4401,200)}), 
                html.P("Filter by Longitude:"),
                dcc.RangeSlider(id='range-slider-1', min=-90, max=-20, step=11, value=[-90,-20],
                                marks={i: '{}'.format(i) for i in range(-90,-19,5)} ), 
                html.P("Filter by Latitude:"),
                dcc.RangeSlider(id='range-slider-2', min=-60, max=20, step=11, value=[-60, 20],
                                marks={i: '{}'.format(i) for i in range(-60,21,5)}), 
                html.P("Filter by Band Width (kHz):"),
                dcc.RangeSlider(id='range-slider-3', min=0, max=10, step=11, value=[0, 10],
                                marks={i: '{}'.format(i) for i in range(0,11,1)} ), 
                html.P("Filter by Trill Rate (Hz):"),
                dcc.RangeSlider(id='range-slider-4', min=0, max=70, step=11, value=[0, 70], 
                                marks={i: '{}'.format(i) for i in range(0,71,5)} ), 
                ])

            @app.callback(
                Output("scatter-plot", "figure"), 
                [Input("range-slider", "value"), Input("range-slider-1", "value"), Input("range-slider-2", "value"),
                 Input("range-slider-3", "value"), Input("range-slider-4", "value")])
            def update_bar_chart(alt_range, lon_range, lat_range, bw_range, tr_range):
                low_alt, high_alt = alt_range
                low_lon, high_lon = lon_range
                low_lat, high_lat = lat_range
                low_bw,  high_bw  = bw_range
                low_tr,  high_tr  = tr_range
                mask =   (all_rates['Altitude'] >= low_alt)  & (all_rates['Altitude'] <= high_alt)  \
                       & (all_rates["Trill Rate"] >= low_tr) & (all_rates["Trill Rate"] <= high_tr) \
                       & (all_rates["Band Width"] >= low_bw) & (all_rates["Band Width"] <= high_bw) \
                       & (all_rates["Longitude"] >= low_lon) & (all_rates["Longitude"] <= high_lon) \
                       & (all_rates["Latitude"] >= low_lat)  & (all_rates["Latitude"] <= high_lat) 
                
                col = rates[mask]["Country"].unique()
                dark = [self.colores[k][2] for k in col]
                light = [self.colores[k][1] for k in col]
                
                podos_df = pd.DataFrame({"x":x, "y":y_podos(x), "slope":[round(m,2)]*len(x), "intercept":[round(b,2)]*len(x)})
                fig = px.line(podos_df, x="x", y="y",  color_discrete_sequence=["black"]*len(x), 
                              line_dash_sequence=["dash"]*len(x), hover_data=["slope", "intercept"], hover_name=["Podos"]*len(x) )
                
                if lines:
                    x_all, y_all = all_rates[mask]["Trill Rate"], all_rates[mask]["Band Width"]
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x_all, y_all)
                    df_all = pd.DataFrame({"x":x, "y":slope*x+intercept, "slope":[round(slope,3)]*len(x), 
                                        "intercept":[round(intercept,3)]*len(x), "p_value":[round(p_value,4)]*len(x), 
                                        "r_value":[round(r_value,4)]*len(x), "std_err":[round(std_err,4)]*len(x)})
                    fig.add_traces(list(px.line(df_all, x="x", y="y",color_discrete_sequence = ["black"]*len(x),
                                                hover_name=["All Data"]*len(x), hover_data=["slope", "intercept","p_value", "std_err","r_value"]
                                                ).select_traces() ))
                    for i in range(len(col)):
                        mask1 = (all_rates['Country'] == col[i]) & mask
                        x0, y0 = all_rates[mask1]["Trill Rate"], all_rates[mask1]["Band Width"]
                        
                        if len(x0)>0:
                            slope, intercept, r_value, p_value, std_err = stats.linregress(x0,y0)
                            df = pd.DataFrame({"x":x, "y":slope*x+intercept, "slope":[round(slope,3)]*len(x), "Country":[col[i]+"_line"]*len(x),
                                                "intercept":[round(intercept,3)]*len(x), "p_value":[round(p_value,4)]*len(x), 
                                                "r_value":[round(r_value,4)]*len(x), "std_err":[round(std_err,4)]*len(x)})
                            
                            fig.add_traces(list(px.line(df, x="x", y="y",color_discrete_sequence = [self.colores[col[i]][1]]*len(x), color="Country",
                                                hover_name=[col[i]+" - Line"]*len(x), hover_data=["slope", "intercept","p_value", "std_err","r_value"],
                                                labels=[col[i]+" - Line"]*len(x)
                                                ).select_traces() ))
                max_points = MaxPoints(all_rates[mask])
                x_all, y_all = list(max_points["Trill Rate"]), list(max_points["Band Width"])
                
                if maxponits:
                    fig.add_traces(list(px.scatter(max_points, x="Trill Rate", y="Band Width", #color="Country", 
                                        color_discrete_sequence=["black"]*len(max_points), symbol_sequence="1",
                                        hover_data=["Filename","Altitude","State"]).select_traces() ) )
                    for i in range(len(col)):
                        mask1 = (all_rates['Country'] == col[i]) & mask
                        points_by_country = MaxPoints(all_rates[mask1])
                        points_by_country["Country_sup"] = points_by_country["Country"]+"_sup"
                        x0, y0 = list(points_by_country["Trill Rate"]), list(points_by_country["Band Width"])
                        fig.add_traces(list(px.scatter(points_by_country, x="Trill Rate", y="Band Width", color="Country_sup", 
                                        color_discrete_sequence=[self.colores[col[i]][2]]*len(points_by_country), 
                                        symbol_sequence="1",
                                        hover_data=["Filename","Altitude","State"]).select_traces() ) )
                if maxlines:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x_all, y_all)
                    df_all = pd.DataFrame({"x":x, "y":slope*x+intercept, "slope":[round(slope,3)]*len(x), 
                                        "intercept":[round(intercept,3)]*len(x), "p_value":[round(p_value,4)]*len(x), 
                                        "r_value":[round(r_value,4)]*len(x), "std_err":[round(std_err,4)]*len(x)})
                    fig.add_traces(list(px.line(df_all, x="x", y="y", color_discrete_sequence = ["black"]*len(x),
                                                hover_name=["All Data"]*len(x), hover_data=["slope", "intercept","p_value", "std_err","r_value"]
                                                ).select_traces() ))
                    for i in range(len(col)):
                        mask1 = (all_rates['Country'] == col[i]) & mask
                        points_by_country = MaxPoints(all_rates[mask1])
                        x0, y0 = list(points_by_country["Trill Rate"]), list(points_by_country["Band Width"])
                        if len(x0)>0:
                            slope, intercept, r_value, p_value, std_err = stats.linregress(x0,y0)
                            df = pd.DataFrame({"x":x, "y":slope*x+intercept, "slope":[round(slope,3)]*len(x), "Country":[col[i]+"_mxline"]*len(x),
                                                "intercept":[round(intercept,3)]*len(x), "p_value":[round(p_value,4)]*len(x), 
                                                "r_value":[round(r_value,4)]*len(x), "std_err":[round(std_err,4)]*len(x)})
                            
                            fig.add_traces(list(px.line(df, x="x", y="y", color="Country",
                                                color_discrete_sequence = [self.colores[col[i]][1]]*len(x),
                                                hover_name=[col[i]+" - Line"]*len(x), hover_data=["slope", "intercept","p_value", "std_err","r_value"],
                                                labels=[col[i]+" - Line"]*len(x)
                                                ).select_traces() ))
                            
                if shapes:
                    for i in range(len(col)):
                        mask1 = (all_rates['Country'] == col[i]) & mask
                        x0, y0 = all_rates[mask1]["Trill Rate"], all_rates[mask1]["Band Width"]
                        fig.add_shape(type="circle", xref="x", yref="y", opacity=0.25,
                                    x0=min(x0), x1=max(x0), y0=min(y0), y1=max(y0),
                                    fillcolor=self.colores[col[i]][2],
                                    name=col[i], editable=True,
                                    templateitemname=col[i],
                                    #showlegend=True,
                                    #line_color=self.colores[col[i]][2],
                                    visible=True
                                    )
                            
                if points:
                    if error:
                        fig.add_traces(list(px.scatter(all_rates[mask], x="Trill Rate", y="Band Width", color="Country", 
                                                    color_discrete_sequence=dark, #countries["color_dark"],
                                                    error_x="tr_e", error_y="BW_e", symbol_sequence="0",
                                                    hover_data=["Filename","Altitude","State"]).select_traces() ) )
                        for i in range(len(fig.data)):  fig.data[i].error_y.thickness = 1
                    else:
                        fig.add_traces(list(px.scatter(all_rates[mask], x="Trill Rate", y="Band Width", color="Country", 
                                            color_discrete_sequence=dark,symbol_sequence="0",
                                            hover_data=["Filename","Altitude","State"]).select_traces() ) )
                    
                fig.update_layout(legend_title_text='<b>      Country<b>',
                                legend=dict(bordercolor="gray", borderwidth=1,
                                        yanchor="top", y=0.99, xanchor="center", x=1.1),
                                xaxis_title="Rate Trill (Hz)", yaxis_title="Band Width (kHz)", plot_bgcolor='ghostwhite',
                                yaxis_range=ylim, xaxis_range=xlim, width=size[0], height=size[1]
                            )
                return fig

            app.run_server(debug=True)
            plt.show()
            
    def Plotly(self,rates, size=(1200,500), error=True, maxlines=False, lines=False, box=False):
        
        plt.close()
        all_rates = rates.copy()
        all_rates["Band Width"] *=1e-3
        all_rates["BW_e"] *=1e-3

        # PODOS line
        x1, y1 = 5.313432835820896, 6.827004219409282
        x2, y2 = 39.343283582089555, 2.4219409282700424
        m = (y2-y1)/(x2-x1)
        b = y1-m*x1
        def y_podos(x): return m*x+b
        x = np.linspace(0,60)
        #max_altitude = 4300
        def MaxPoints(all_rates):
            trs, bws, c = [], [], 5
            for i in range(20):
                low, high = c*i, c*(i+1)
                mask = (all_rates["Trill Rate"]>=low) & (all_rates["Trill Rate"]<high)
                if len(all_rates[mask])>0:
                    index = all_rates[mask]["Trill Rate"].argmax()
                    tr, bw = all_rates[mask].iloc[index][["Trill Rate", "Band Width"]]
                    trs.append(tr); bws.append(bw)
            return trs, bws

        app = Dash(__name__)
        app.layout = html.Div([
            #html.H4('Interactive scatter plot with Iris dataset'),
            dcc.Graph(id="scatter-plot"),
            html.P("Filter by Altitude (m):"),
            dcc.RangeSlider(id='range-slider', min=-1, max=4200, step=11, value=[0,4200],
                            marks={i: '{}'.format(i) for i in range(0,4401,200)}), 
            html.P("Filter by Longitude:"),
            dcc.RangeSlider(id='range-slider-1', min=-90, max=-20, step=1, value=[-90,-20],
                            marks={i: '{}'.format(i) for i in range(-90,-19,5)} ), 
            html.P("Filter by Latitude:"),
            dcc.RangeSlider(id='range-slider-2', min=-60, max=20, step=1, value=[-60, 20],
                            marks={i: '{}'.format(i) for i in range(-60,21,5)}), 
            html.P("Filter by Band Width (kHz):"),
            dcc.RangeSlider(id='range-slider-3', min=0, max=10, step=11, value=[0, 10],
                            marks={i: '{}'.format(i) for i in range(0,11,1)} ), 
            html.P("Filter by Trill Rate (Hz):"),
            dcc.RangeSlider(id='range-slider-4', min=0, max=70, step=11, value=[0, 70], 
                            marks={i: '{}'.format(i) for i in range(0,71,5)} ), 
            ])

        @app.callback(
            Output("scatter-plot", "figure"), 
            [Input("range-slider", "value"), Input("range-slider-1", "value"), Input("range-slider-2", "value"),
                Input("range-slider-3", "value"), Input("range-slider-4", "value")])
        def update_bar_chart(alt_range, lon_range, lat_range, bw_range, tr_range):
            low_alt, high_alt = alt_range
            low_lon, high_lon = lon_range
            low_lat, high_lat = lat_range
            low_bw,  high_bw  = bw_range
            low_tr,  high_tr  = tr_range
            mask = (all_rates['Altitude']   >= low_alt) & (all_rates['Altitude']   <= high_alt) \
                 & (all_rates["Trill Rate"] >= low_tr)  & (all_rates["Trill Rate"] <= high_tr)  \
                 & (all_rates["Band Width"] >= low_bw)  & (all_rates["Band Width"] <= high_bw)  \
                 & (all_rates["Longitude"]  >= low_lon) & (all_rates["Longitude"]  <= high_lon) \
                 & (all_rates["Latitude"]   >= low_lat) & (all_rates["Latitude"]   <= high_lat) 
            
            col = all_rates[mask]["Country"].unique()
            dark = [self.colores[k][2] for k in col]
            light = [self.colores[k][1] for k in col]
            
            fig = make_subplots(rows=1, cols=3, column_widths=[1.5, 1.8, 0.2])
            
            podos_df = pd.DataFrame({"x":x, "y":y_podos(x), "slope":[round(m,2)]*len(x), "intercept":[round(b,2)]*len(x)})
            fig.add_traces(list(px.line(podos_df, x="x", y="y",  color_discrete_sequence=["black"]*len(x), 
                                        line_dash_sequence=["dash"]*len(x), hover_data=["slope", "intercept"], 
                                        hover_name=["Podos"]*len(x) ).select_traces() ), rows=1, cols=1)
            
            if lines:
                x_all, y_all = all_rates[mask]["Trill Rate"], all_rates[mask]["Band Width"]
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_all, y_all)
                df_all = pd.DataFrame({"x":x, "y":slope*x+intercept, "slope":[round(slope,2)]*len(x), 
                                    "intercept":[round(intercept,2)]*len(x), "p_value":[round(intercept,2)]*len(x), 
                                    "std_err":[round(intercept,2)]*len(x)})
                fig.add_traces(list(px.line(df_all, x="x", y="y",color_discrete_sequence = ["black"]*len(x),
                                            hover_name=["All Data"]*len(x), hover_data=["slope", "intercept","p_value", "std_err"]
                                            ).select_traces() ))
                for i in range(len(col)):
                    mask1 = (all_rates['Country'] == col[i]) & mask
                    x0, y0 = all_rates[mask1]["Trill Rate"], all_rates[mask1]["Band Width"]
                    if len(x0)>0:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x0,y0)
                        df = pd.DataFrame({"x":x, "y":slope*x+intercept, "slope":[round(slope,2)]*len(x), 
                                    "intercept":[round(intercept,2)]*len(x), "p_value":[round(intercept,2)]*len(x), 
                                    "std_err":[round(intercept,2)]*len(x)})
                        fig.add_traces(list(px.line(df, x="x", y="y",color_discrete_sequence = [self.colores[col[i]][1]]*len(x),
                                            hover_name=[col[i]+" - Line"]*len(x), hover_data=["slope", "intercept","p_value", "std_err"],
                                            labels=[col[i]+" - Line"]*len(x)
                                            ).select_traces() ))
                        
            if maxlines:
                x_all, y_all = MaxPoints(all_rates[mask])
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_all, y_all)
                df_all = pd.DataFrame({"x":x, "y":slope*x+intercept, "slope":[round(slope,2)]*len(x), 
                                    "intercept":[round(intercept,2)]*len(x), "p_value":[round(intercept,2)]*len(x), 
                                    "std_err":[round(intercept,2)]*len(x)})
                fig.add_traces(list(px.line(df_all, x="x", y="y",color_discrete_sequence = ["black"]*len(x),
                                            hover_name=["All Data"]*len(x), hover_data=["slope", "intercept","p_value", "std_err"]
                                            ).select_traces() ))
                for i in range(len(col)):
                    mask1 = (all_rates['Country'] == col[i]) & mask
                    x0, y0 = MaxPoints(all_rates[mask])
                    if len(x0)>0:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x0,y0)
                        df = pd.DataFrame({"x":x, "y":slope*x+intercept, "slope":[round(slope,2)]*len(x), 
                                    "intercept":[round(intercept,2)]*len(x), "p_value":[round(intercept,2)]*len(x), 
                                    "std_err":[round(intercept,2)]*len(x)})
                        fig.add_traces(list(px.line(df, x="x", y="y",color_discrete_sequence = [self.colores[col[i]][1]]*len(x),
                                            hover_name=[col[i]+" - Line"]*len(x), hover_data=["slope", "intercept","p_value", "std_err"],
                                            labels=[col[i]+" - Line"]*len(x)
                                            ).select_traces() ))
            
            if error:
                fig.add_traces(list(px.scatter(all_rates[mask], x="Trill Rate", y="Band Width", color="Country", 
                                                color_discrete_sequence=dark, #size=[10.01]*len(all_rates[mask]),#countries["color_dark"],
                                                error_x="tr_e", error_y="BW_e",
                                                symbol_sequence="0",
                                                hover_data=["Filename","Altitude","State"]).select_traces() ),rows=1, cols=1 )
                for i in range(len(fig.data)):  fig.data[i].error_y.thickness = 1
            else:
                fig.add_traces(list(px.scatter(all_rates[mask], x="Trill Rate", y="Band Width", color="Country", color_discrete_sequence=dark,
                                               symbol_sequence="0", hover_data=["Filename","Altitude","State"]).select_traces() ),rows=1, cols=1 )
            for i in range(len(fig.data)):  fig.data[i].error_y.thickness = 1
            
            fig.update_traces(marker_size=6)
            fig.update_layout(legend_title_text='<b>      Country<b>',
                            legend=dict(bordercolor="gray", borderwidth=1,
                                    yanchor="top", y=0.99, xanchor="center", x=0.7),
                            xaxis_title="Rate Trill (Hz)", yaxis_title="Band Width (kHz)", plot_bgcolor='ghostwhite',
                            yaxis_range=[0,10], xaxis_range=[0,70], width=size[0], height=size[1] )
            
            
            fig.add_traces(list(px.choropleth(locations=col, locationmode="country names", scope="world", color=col,
                                              color_discrete_sequence=light, width=size[0], height=size[1]).select_traces()))
            
            fig.update_traces(showlegend=False)
            N, n = 1000, 3
            lon = np.linspace(min(all_rates[mask]["Longitude"])-n, max(all_rates[mask]["Longitude"])+n, N)
            lat = np.linspace(min(all_rates[mask]["Latitude"])-n, max(all_rates[mask]["Latitude"])+n, N)
            fig.add_traces(list(px.scatter_geo(all_rates[mask], lon="Longitude", lat="Latitude", symbol_sequence="x",
                                                hover_name="Filename", hover_data=["Longitude", "Latitude", "Altitude"],
                                                color="Country", color_discrete_sequence=dark ).select_traces(),
                                ))
            if box:
                fig.add_traces(list(px.scatter_geo(lon=lon, lat=[lat[0]]*N, color_discrete_sequence=["black"]*N, symbol_sequence=["square"]*N, opacity=0.5 ).select_traces() ))
                fig.add_traces(list(px.scatter_geo(lon=lon, lat=[lat[-1]]*N, color_discrete_sequence=["black"]*N, symbol_sequence=["square"]*N, opacity=0.5 ).select_traces() ))
                fig.add_traces(list(px.scatter_geo(lon=[lon[0]]*N, lat=lat, color_discrete_sequence=["black"]*N, symbol_sequence=["square"]*N, opacity=0.5 ).select_traces() ))
                fig.add_traces(list(px.scatter_geo(lon=[lon[-1]]*N, lat=lat, color_discrete_sequence=["black"]*N, symbol_sequence=["square"]*N, opacity=0.5 ).select_traces() ))
            fig.update_geos(center=dict(lon=-60, lat=-14), landcolor="whitesmoke", showcountries=True, #showrivers=True,
                            #lataxis_range=[-14+lat_range[0],-14-lat_range[1]], 
                            #lonaxis_range=[-60+lon_range[0],-60-lon_range[1]])
                            lataxis_range=[-50,40], lonaxis_range=[-40, 40],)
            
            return fig

        app.run_server(debug=True)
        plt.show()
        
    #%%
    def RatesBW3D(self, rates, xlim=(0,70), ylim=(0,10), zlim=(0,4200), size=(1000,500),
                  error = False):

        app = Dash(__name__)
        app.layout = html.Div([
            #html.H4('Interactive scatter plot with Iris dataset'),
            dcc.Graph(id="scatter-plot"),
            html.P("Filter by Longitude:"),
            dcc.RangeSlider(id='range-slider-1', min=-90, max=-20, step=11, value=[-90,-20],
                            marks={i: '{}'.format(i) for i in range(-90,-19,5)} ), 
            html.P("Filter by Latitude:"),
            dcc.RangeSlider(id='range-slider-2', min=-60, max=20, step=11, value=[-60, 20],
                            marks={i: '{}'.format(i) for i in range(-60,21,5)}), 
            
            ])

        @app.callback(
            Output("scatter-plot", "figure"), 
            [Input("range-slider-1", "value"), Input("range-slider-2", "value")])

        def update_bar_chart(lon_range, lat_range):
            low_lon, high_lon = lon_range
            low_lat, high_lat = lat_range
            
            all_rates = rates.copy()
            all_rates["Band Width"] *= 1e-3
            all_rates["BW_e"] *= 1e-3
            mask =  (all_rates["Longitude"] >= low_lon) & (all_rates["Longitude"] <= high_lon) \
                    & (all_rates["Latitude"] >= low_lat)  & (all_rates["Latitude"] <= high_lat) 
                    
            col = all_rates[mask]["Country"].unique()
            dark = [self.colores[k][2] for k in col]
            #light = [self.colores[k][1] for k in col]

            if error:
                fig = px.scatter_3d(all_rates[mask], x='Trill Rate', y='Band Width', z='Altitude', color="Country", 
                                symbol_sequence=["circle"]*len(all_rates), 
                                error_x="trill_time_mean_e", error_y="BW_e",
                                hover_name="Country", hover_data=["Country","State","Filename"],
                                opacity=0.8, color_discrete_sequence=dark,
                                range_x=xlim, range_y=ylim, range_z=zlim )
            else: 
                fig = px.scatter_3d(all_rates[mask], x='Trill Rate', y='Band Width', z='Altitude', color="Country", 
                                    symbol_sequence=["circle"]*len(all_rates), 
                                    hover_name="Country", hover_data=["Country","State","Filename"],
                                    opacity=0.8, color_discrete_sequence=dark,
                                    range_x=xlim, range_y=ylim, range_z=zlim )
            fig.update_layout(showlegend=True, height=size[1], width=size[0], 
                              legend=dict(title="<b>     Country<b>", bordercolor="gray", borderwidth=1,
                                        yanchor="top", y=0.85, xanchor="left", x=0.75),
                              scene=dict(xaxis_title='Trill Rate (Hz)',
                                         yaxis_title='Band Width (Hz)',
                                         zaxis_title='Altitude (m)',
                                        #xaxis_autorange="reversed", yaxis_autorange="reversed" 
                                        ),
                            )
            fig.update_traces(marker_size = 4)
            return fig

        app.run_server(debug=True)
    #%%
    def RatesBW(self, rates, error=False, size=(1000,600), xlim=(0,4200), ylim1=(0,10), ylim2=(0,70)):
        app = Dash(__name__)
        app.layout = html.Div([
            #html.H4('Interactive scatter plot with Iris dataset'),
            dcc.Graph(id="scatter-plot"),
            html.P("Filter by Longitude:"),
            dcc.RangeSlider(id='range-slider-1', min=-90, max=-20, step=11, value=[-90,-20],
                            marks={i: '{}'.format(i) for i in range(-90,-19,5)} ), 
            html.P("Filter by Latitude:"),
            dcc.RangeSlider(id='range-slider-2', min=-60, max=20, step=11, value=[-60, 20],
                            marks={i: '{}'.format(i) for i in range(-60,21,5)}), 
            ])

        @app.callback(
            Output("scatter-plot", "figure"), 
            [Input("range-slider-1", "value"), Input("range-slider-2", "value")])

        def update_bar_chart(lon_range, lat_range):
            low_lon, high_lon = lon_range
            low_lat, high_lat = lat_range
            
            all_rates = rates.copy()
            all_rates["Band Width"] *= 1e-3
            all_rates["BW_e"] *= 1e-3
                    
            mask =  (all_rates["Longitude"] >= low_lon) & (all_rates["Longitude"] <= high_lon) \
                    & (all_rates["Latitude"] >= low_lat)  & (all_rates["Latitude"] <= high_lat) 
                
            col = all_rates[mask]["Country"].unique()
            dark = [self.colores[k][2] for k in col]
            light = [self.colores[k][1] for k in col]

            fig = plotly.subplots.make_subplots(rows=2, cols=1, specs=[[{"type": "xy"}], [{"type": "xy"}]],
                                                shared_xaxes=True, vertical_spacing=0.02)
            if error:
                fig.add_traces(list(px.scatter(all_rates[mask], x="Altitude", y="Band Width", color="Country", 
                                               error_y="BW_e",
                                               color_discrete_sequence=dark,symbol_sequence="0",
                                               hover_data=["Filename","Altitude","Trill Rate","State"],).select_traces() ), rows=1, cols=1 )
                for trace in fig['data']:  trace['showlegend'] = False
                fig.add_traces(list(px.scatter(all_rates[mask], x="Altitude", y="Trill Rate", color="Country", custom_data="",
                                               error_y="tr_e",
                                               color_discrete_sequence=dark,symbol_sequence="0",
                                               hover_data=["Filename","Altitude","Band Width","State"]).select_traces() ), rows=2, cols=1 )
            else:
                fig.add_traces(list(px.scatter(all_rates[mask], x="Altitude", y="Band Width", color="Country", 
                                                    color_discrete_sequence=dark,symbol_sequence="0",
                                                    hover_data=["Filename","Altitude","Trill Rate","State"],).select_traces() ), rows=1, cols=1 )
                for trace in fig['data']:  trace['showlegend'] = False

                fig.add_traces(list(px.scatter(all_rates[mask], x="Altitude", y="Trill Rate", color="Country", custom_data="",
                                                        color_discrete_sequence=dark,symbol_sequence="0",
                                                        hover_data=["Filename","Altitude","Band Width","State"]).select_traces() ), rows=2, cols=1 )
            fig['layout']['yaxis1']['title']='Band Width (Hz)'
            fig['layout']['yaxis2']['title']='Trill Rate (Hz)'
            fig['layout']['xaxis2']['title']='Altitude (m)'

            fig.update_layout(xaxis_range=xlim, yaxis1=dict(range=ylim1), yaxis2=dict(range=ylim2), 
                              showlegend=True, height=size[1], width=size[0],
                              legend=dict(title="<b>     Country<b>", bordercolor="gray", 
                                          borderwidth=1, yanchor="top", y=0.99, xanchor="left", x=1.05) )
            return fig

        app.run_server(debug=True)
    #%%
    def Counter(self, rates):
        val = rates.pivot_table(index = ['Country'], aggfunc ='size')

        dark = [self.colores[k][2] for k in val.index]
        df = pd.DataFrame({"Country":val.index, "Number":val, "Color":dark})

        fig = px.pie(df, names="Country", color="Country", color_discrete_sequence=dark, values="Number", opacity=0.75, 
                    hover_name="Country", #hover_data=["Country"],
                    hole=.3)
        fig.update_traces(textinfo='value+percent')
        fig.update_layout(legend=dict(title="<b>      Country<b>", bordercolor="gray", borderwidth=1,
                                    yanchor="top", y=0.90, xanchor="left", x=1.2),
                        annotations=[dict(text=str(sum(val)), x=0.5, y=0.5, font_size=20, showarrow=False)],
                        width=600, height=500)

        fig.show()
        
    #%%
    def SelectData(self, obj, FF_on=False, xlim=None, figsize=None, waveform=False, save=None): 
                
        ticks = ticker.FuncFormatter(lambda x, pos: '{:g}'.format(x*1e-3))
        ticks_x = ticker.FuncFormatter(lambda x, pos: '{:.3f}'.format(x+obj.t0))

        if figsize is None: figsize=(8*self.figsize[0]//5, self.figsize[1]*5/3)
        if xlim is None:    xlim = (obj.time[0], obj.time[-1]) #(xlim[0]-obj.t0, xlim[1]-obj.t0)
        #else:                xlim = (obj.time[0], obj.time[-1])

        plt.close()

        if waveform:
            fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True, gridspec_kw={'height_ratios': [1,2]})
            plt.subplots_adjust(right=0.8)
            ax = axes[1]
            
            axes[0].plot(obj.time_s, obj.s,'k', label='waveform')
            axes[0].plot(obj.time_s, np.ones(obj.time_s.size)*obj.umbral, '--', label='umbral')
            axes[0].plot(obj.time_s, obj.envelope, label='envelope')
            axes[0].legend(bbox_to_anchor=(1.01, 0.95))
            axes[0].xaxis.set_major_formatter(ticks_x)
            axes[0].set_ylim((0,1))
            axes[0].set_ylabel('Amplitude (a.u)'); axes[0].set_xlabel(''); 

        else:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            plt.subplots_adjust(right=0.8)

        img = Specshow(obj.Sxx_dB, x_axis="s", y_axis="linear", sr=obj.fs,
                        hop_length=obj.hop_length, ax=ax, cmap=self.cmap)

        if FF_on: 
            if obj.ff_method=="yin" or obj.ff_method=="pyin" or obj.ff_method=="manual":
                ax.plot(obj.time, obj.FF,  "bo", label=r"FF$_{{}}$".format(obj.ff_method), ms=8)
            elif obj.ff_method=="both":
                ax.plot(obj.time, obj.FF,  "co", label=r"FF$_{pyin}$", ms=8)
                ax.plot(obj.time, obj.FF2, "b*", label=r"FF$_{yin}$", ms=8)
            #ax.legend()
            ax.legend(bbox_to_anchor=(1.01, 0.65))

        ax.yaxis.set_major_formatter(ticks)
        ax.xaxis.set_major_formatter(ticks_x)
        ax.set_ylim(obj.flim); ax.set_xlim(xlim)
        ax.set_ylabel('Frequency (kHz)'); ax.set_xlabel('Time (s)');

        fig.suptitle("Audio Sample: "+obj.file_name, fontsize=18)
        self.klicker = Klicker_Multiple(fig, ax)
        fig.tight_layout()

        plt.show()
        
        path_save = obj.paths.images / "{}-AllSongAndSyllable.png".format(obj.file_name[:-4])
        if save is None: save = self.save
        if save: fig.savefig(path_save, transparent=True, bbox_inches='tight')