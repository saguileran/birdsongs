from .utils import *

class Ploter(object): 
    def __init__(self, save=False, cmap="magma"): #" gray_r afmhot_r"
        self.save = save
        self.cmap = cmap
    
    def PlotAlphaBeta(self, obj, xlim=(-0.05,.2), ylim=(-0.2,0.9)):
        if "synth" in obj.id:
            if obj.alpha.max()>0.2: xlim=(-0.05,1.1*obj.alpha.max())
            if obj.beta.max()>0.9:  ylim=(-0.2,1.1*obj.beta.max())
            fig = plt.figure(constrained_layout=True, figsize=(10, 6))
            gs  = GridSpec(2, 2, figure=fig)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[1:, 0])
            ax3 = fig.add_subplot(gs[:, 1])

            fig.suptitle("GridSpec")
            fig.tight_layout(pad=3.0) #fig.subplots_adjust(top=0.85)

            viridis = mpl.colormaps["Blues"] #cm.get_cmap('Blues')
            c       = viridis(np.linspace(0.3, 1, np.size(obj.time_s)))

            ax1.scatter(obj.time_s, obj.alpha, c=c, label="alfa")
            ax1.set_title('Air-Sac Pressure')
            ax1.grid()
            ax1.set_ylabel('α (a.u.)'); #ax1.set_xlabel('Time (s)'); 
            ax1.set_ylim(xlim);        #ax[0].legend()

            ax2.scatter(obj.time_s, obj.beta, c=c, label="beta")
            ax2.set_title('Labial Tension')
            ax2.grid()
            ax2.set_xlabel(r'Time (s)'); ax2.set_ylabel('β (a.u.)')
            ax2.set_ylim(ylim);         ax2.sharex(ax1);        #ax[0].legend()

            # ------------- Bogdanov–Takens bifurcation ------------------
            ax3.scatter(obj.alpha, obj.beta, c=c, label="Parameters", marker="_")
            ax3.plot(-1/27, 1/3, 'ko')#, label="Cuspid Point"); 
            ax3.axvline(0, color='red', lw=1)#, label="Hopf Bifurcation")
            ax3.plot(obj.mu1_curves[0], obj.beta_bif, '-g', lw=1)#, label="Saddle-Noddle\nBifurcation"); 
            ax3.text(-0.02,0.6, "Hopf",rotation=90, color="r")
            ax3.text(-0.04,0.39,"CP",  rotation=0,  color="k")
            ax3.text(-0.03,0.02,  "SN",  rotation=0,  color="g")
            ax3.text(0.1, 0.005, "SN",  rotation=0,  color="g")

            ax3.plot(obj.mu1_curves[1], obj.beta_bif, '-g', lw=1)
            ax3.fill_between(obj.mu1_curves[1], obj.beta_bif, 10, where=obj.mu1_curves[1] > 0, color='gray', alpha=0.25)#, label='BirdSongs')
            ax3.set_ylabel('Tension (a.u.)'); ax3.set_xlabel('Pressure (a.u.)')
            ax3.set_title('Parameter Space')
            ax3.legend()
            ax3.set_xlim(xlim); ax3.set_ylim(ylim)
            fig.suptitle("Air-Sac Pressure (α) and Labial Tension (β) Parameters", fontsize=20)#, family='fantasy')
            plt.show()
            
            return fig, gs

            if self.save: fig.savefig(obj.paths.results+"MotorGesturesParameters-{}-{}-{}.png".format(obj.id,obj.no_file,obj.no_syllable))
    
        else: 
            print("This  is not a synthetic object, try with otherone.")
    
    
    def PlotVs(self,obj, xlim=(0,0.025)):
        if "synth" in obj.id:
            fig, ax = plt.subplots(3, 1, figsize=(12, 9))
            fig.subplots_adjust(wspace=0.35, hspace=0.4)

            #time = obj.time[:obj.Vs.shape[0]]

            ax[0].plot(obj.timesVs, obj.Vs[:,4], color='b')
            #ax[0].set_xlim((0,1e5))
            ax[0].set_xlabel("time (s)"); ax[0].set_ylabel("$P_{out}$");
            ax[0].set_title("Trachea Output Pressure")

            ax[1].plot(obj.timesVs, obj.Vs[:,1], color='g') 
            #ax[1].set_xlim(xlim)
            ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("$P_{in}$");
            ax[1].set_title("Trachea Input Pressure")

            ax[2].plot(obj.timesVs, obj.Vs[:,0], color='r')
            #ax[2].set_xlim(xlim)
            ax[2].set_xlabel("time (s)"); ax[2].set_ylabel("$x(t)$");
            ax[2].set_title("Labial position")
            ax[2].sharex(ax[1])

            fig.suptitle('Labial Parameters (vector $v$)', fontsize=20)
            plt.show()
            
            return fig, ax

            if self.save: fig.savefig(obj.paths.results+"MotorGesturesVariables-{}-{}-{}.png".format(obj.id,obj.no_file,obj.no_syllable)) 
            
        else:  print("This is not a synthetic object")
            
    
    
    def Plot(self, obj, syllable_on=False, chunck_on=False, FF_on=False): 
        ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x*1e-3))
        if "song" in obj.id:
            fig, ax = plt.subplots(2+int(syllable_on), 1, figsize=(12, 6+3*int(syllable_on)))
            fig.subplots_adjust(hspace=0.4, wspace=0.4)

            img = librosa.display.specshow(obj.Sxx_dB, x_axis="s", y_axis="linear", sr=obj.fs,
                         hop_length=obj.hop_length, ax=ax[0], cmap=self.cmap)
            
            if FF_on: 
                ax[0].plot(obj.time, obj.FF, "co", label="Fundamental Frequency", ms=8)
                ax[0].legend()
            # for i in range(len(obj.syllables)):  
            #     ax[0].plot([obj.time[obj.syllables[i][0]], obj.time[obj.syllables[i][-1]]], [0, 0], 'k', lw=5)
            #     ax[0].text((obj.time[obj.syllables[i][-1]]-obj.time[obj.syllables[i][0]])/2, 0.5, str(i))
            ax[0].set_ylim(obj.flim); ax[0].set_xlim(min(obj.time), max(obj.time));
            ax[0].set_title("Song Spectrum"); 
            ax[0].yaxis.set_major_formatter(ticks)
            ax[0].set_ylabel('f (kHz)'); #ax[0].set_xlabel('t (s)'); 


            ax[1].plot(obj.time_s, obj.s,'k', label='audio')
            ax[1].plot(obj.time_s, np.ones(obj.time_s.size)*obj.umbral, '--', label='umbral')
            ax[1].plot(obj.time_s, obj.envelope, label='envelope')
            ax[1].legend(loc='upper right')#loc=1, title='Data')
            ax[1].set_title("Song Sound Wave")
            ax[1].set_xlabel('time (s)'); ax[1].set_ylabel('Amplitud normalaized');
            ax[1].sharex(ax[0])

            if chunck_on:
                ax[0].plot(obj.chunck.time+obj.chunck.t0, obj.chunck.FF, 'gv', label='Chunck', ms=10)
                ax[2].plot(obj.chunck.time+obj.chunck.t0-obj.syllable.t0, obj.chunck.FF, 'gv', label='Chunck', ms=8)

            if syllable_on:
                ax[0].plot(obj.syllable.time+obj.syllable.t0, obj.syllable.FF, 'b+', label='Syllable'.format(obj.syllable.fs), ms=6)

                img = librosa.display.specshow(obj.syllable.Sxx_dB, x_axis="s", y_axis="linear", sr=obj.syllable.fs,
                         hop_length=obj.syllable.hop_length, ax=ax[2], cmap=self.cmap)
                
                ax[2].plot(obj.syllable.time, obj.syllable.FF, 'b+', label='Syllable', ms=15)
                ax[2].set_ylim(obj.flim); 
                ax[2].set_xlim((obj.syllable.time[0], obj.syllable.time[-1]));
                ax[2].legend(loc='upper right', title="FF")
                ax[2].set_xlabel('t (s)'); ax[2].set_ylabel('f (khz)')
                ax[2].set_title('Single Syllable Spectrum, No {}'.format(obj.no_syllable))
                ax[2].yaxis.set_major_formatter(ticks)
                
                ax[0].legend(loc='upper right', title="FF")

                path_save = obj.paths.results / "AllSongAndSyllable-{}-{}.png".format(obj.no_file,obj.no_syllable)
            else: path_save = obj.paths.results / "AllSongAndSyllable-{}.png".format(obj.no_file)

            fig.suptitle('Audio: {}'.format(obj.file_name.name), fontsize=18)
            plt.show()
            return fig, ax
        
            if self.save: fig.savefig(path_save)
        else:  # syllable ------------------------------------------------
            fig, ax = plt.subplots(2, 1, figsize=(12, 6+3*int(syllable_on)))
            fig.subplots_adjust(hspace=0.4, wspace=0.4)

            img = librosa.display.specshow(obj.Sxx_dB, x_axis="s", y_axis="linear", sr=obj.fs,
                         hop_length=obj.hop_length, ax=ax[0])
            #fig.colorbar(img, ax=ax[0], format="%+2.f dB")
            ax[0].plot(obj.time, obj.FF)
            ax[0].yaxis.set_major_formatter(ticks)
            ax[0].plot(obj.time, obj.FF,"bo", ms=7,label="Fundamental Frequency")
            ax[0].legend()
            ax[0].set_ylim(obj.flim); ax[0].set_xlim(min(obj.time), max(obj.time));
            ax[0].set_title("Spectrum"); 
            ax[0].set_ylabel('f (kHz)'); #ax[0].set_xlabel('t (s)'); 


            ax[1].plot(obj.time_s, obj.s,'k', label='audio')
            ax[1].plot(obj.time_s, obj.envelope, label='envelope')
            ax[1].legend(loc='upper right')#loc=1, title='Data')
            ax[1].set_title("Sound Wave")
            ax[1].set_xlabel('t (s)'); ax[1].set_ylabel('Amplitud normalaized');
            ax[1].sharex(ax[0])

            fig.suptitle('id: {}, NoFile: {}, NoSyllable: {}'.format(obj.id, obj.no_file, obj.no_syllable), fontsize=16)
            plt.show()

            path_save = obj.paths.results+"AllSongAndSyllable-{}-{}.png".format(obj.no_file,obj.no_syllable)
            
            return fig, ax
            if self.save: fig.savefig(path_save)

        
    def Syllables(self, obj, obj_synth): #PlotSyllables
        fig, ax = plt.subplots(2, 2, figsize=(12, 5), sharex=True, sharey='col')
        fig.subplots_adjust(top=0.85)     

        ax[0,0].plot(obj.time_s, obj.s, label='canto', c='b')
        ax[0,0].set_title('Real')
        ax[0,0].plot(obj.time_s, obj.envelope, label='envelope', c='k')
        ax[0,0].legend(); ax[0,0].set_ylabel("Amplitud (a.u.)")
        ax[1,0].plot(obj_synth.time_s, obj_synth.s, label='synthetic', c='g')
        ax[1,0].set_title('Synthetic') 
        ax[1,0].plot(obj_synth.time_s, obj_synth.envelope
    , label='envelope', c='k')
        ax[1,0].legend(); ax[1,0].set_xlabel('t (s)'); ax[1,0].set_ylabel("Amplitud (a.u.)")

        Delta_tu   = obj.time[-1] - obj.time[0]
        Delta_tu_s = 1#tu_s[-1] - tu_s[0]

        img = librosa.display.specshow(obj.Sxx_dB, x_axis="s", y_axis="linear", sr=obj.fs,
                         hop_length=obj.hop_length, ax=ax[0,1], cmap=self.cmap)
        fig.colorbar(img, ax=ax[0,1], format="%+2.f dB")
        #C = img.get_array().data.reshape(obj.Sxx_dB.shape)
        #ax[0,1].pcolormesh(obj.times, obj.freqs*1e-3, C, cmap=plt.get_cmap(self.cmap), rasterized=True, edgecolors="face")#, vmin=10, vmax=70)
        #fig.colorbar(pcm, ax=ax[0,1], location='right', label='Power (dB)')
        
        ax[0,1].plot(obj.time, obj.FF, 'bo-', lw=2)
        ax[0,1].set_title('Real'); ax[0,1].set_ylabel('f (khz)'); ax[0,1].set_ylim(obj.flim);
        
        img = librosa.display.specshow(obj_synth.Sxx_dB, x_axis="s", y_axis="linear", sr=obj_synth.fs,
                         hop_length=obj_synth.hop_length, ax=ax[1,1], cmap=self.cmap)
        fig.colorbar(img, ax=ax[1,1], format="%+2.f dB")
        #C = img.get_array().data.reshape(obj_synth.Sxx_dB.shape)
        #ax[1,1].pcolormesh(obj_synth.times, obj_synth.freqs*1e-3,  C, cmap=plt.get_cmap(self.cmap), rasterized=True, edgecolors="face")#, vmin=10, vmax=70)
        #fig.colorbar(pcm, ax=ax[1,1], location='right', label='Power (dB)')
        
        ax[1,1].plot(obj_synth.time, obj_synth.FF, 'go-', lw=2)
        ax[1,1].set_title('Synthetic') 
        ax[1,1].set_ylim(obj.flim);   ax[1,1].set_xlim(min(obj_synth.time), max(obj_synth.time))
        ax[1,1].set_xlabel('t (s)'); ax[1,1].set_ylabel('f (khz)');

        #fig.tight_layout(); 
        fig.suptitle('Sound Waves and Spectrograms', fontsize=20)
        plt.show()
        return fig, ax
        if self.save: fig.savefig(obj.paths.results+"SoundAndSpectros-{}-{}-{}.png".format(obj.id,obj.no_file,obj.no_syllable))

    
    def Result(self, obj, obj_synth, cmp="afmhot_r"):
        if not("synth" in obj.id) and "synth" in obj_synth.id:
            ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x*1e-3))
            fig = plt.figure(constrained_layout=False, figsize=(24, 12))
            gs  = fig.add_gridspec(nrows=4, ncols=5, wspace=0.05, hspace=0.05, width_ratios=[1,1,1,1.2,1.2], left=0.05, right=0.98,)
            vmin, vmax = obj.Sxx_dB.min(), obj.Sxx_dB.max()
            # ----- FF ---------------
            ax1 = fig.add_subplot(gs[0:2, :3])
            
            img = librosa.display.specshow(obj.Sxx_dB, x_axis="s", y_axis="linear", sr=obj.fs,
                         hop_length=obj.hop_length, ax=ax1, cmap=self.cmap)
            fig.colorbar(img, ax=ax1, format="%+2.f dB")
            
            ax1.plot(obj.time,       obj.f_msf,       'D-', color="skyblue", label=r'$FF_{msf}$ real',ms=12)
            ax1.plot(obj_synth.time, obj_synth.f_msf, 'X-', color="lightgreen",label=r'$FF_{msf}$ synt', ms=12)

            ax1.plot(obj_synth.time, obj_synth.centroid, 'p-', color="olive",  label=r'$F_{cent}$ synth', ms=12)
            ax1.plot(obj.time,       obj.centroid,       '+-', color="yellow", label=r'$F_{cent}$ real', ms=12)
            
            ax1.plot(obj.time,        obj.FF,       'b*-', label=r'FF real',ms=25)
            ax1.plot(obj_synth.time,  obj_synth.FF, 'go-', label=r'FF synt', ms=12)

            ax1.legend(borderpad=0.6, labelspacing=0.7, title="Feature"); ax1.set_ylim(obj.flim); 
            ax1.set_xlim((obj.time[0], obj.time[-1]))
            ax1.set_ylabel('f (khz)'); ax1.set_xlabel('time (s)');
            ax1.yaxis.set_major_formatter(ticks)
            ax1.set_title('Spectrogram - Fundamental Frequency (FF)')
            
            ax2 = fig.add_subplot(gs[0:2, 3:])
            ax2.plot(obj_synth.time,  obj_synth.deltaFF ,      "*-", color="k",  ms=12, lw=3, label=r' $||ΔFF||_{}$= {:.4f}, mean={:.4f}'.format(obj_synth.ord, obj_synth.scoreFF, obj_synth.deltaFF_mean)); 
            ax2.plot(obj_synth.time, obj_synth.deltaRMS,      "-p", color="r",  label=r' $|| ΔF_{{ rms }}||_{}$= {:.4f}, mean={:.4f}'.format(obj_synth.ord,  obj_synth.scoreRMS, obj_synth.scoreRMS_mean)); 
            ax2.plot(obj.time,       obj_synth.deltaCentroid,  "-o", color="y", label=r'$ || \Delta F_{{ centroid }}||_{}$ = {:.4f}, mean={:.4f}'.format(obj_synth.ord, obj_synth.scoreCentroid, obj_synth.scoreCentroid_mean)); 
            ax2.plot(obj.time,       obj_synth.deltaF_msf,     "D-", color="purple", label=r'$ || \Delta F_{{ msf }}||_{}$ = {:.4f}, mean={:.4f}'.format(obj_synth.ord, obj_synth.scoreF_msf, obj_synth.scoreF_msf_mean)); 


            ax2.plot(obj_synth.time, obj_synth.rms*1e-3, 'p-', color="darkred", label=r'$F_{rms}$ synth', ms=7)
            ax2.plot(obj.time,       obj.rms*1e-3,       'rv-', label=r'$F_{rms}$ real', ms=9)

            ax2.set_xlabel('time (s)'); ax2.set_ylabel('f (kHz)'); ax2.legend(title="Features")
            ax2.set_title('Fundamental Frequency Error (ΔFF)'); 
            if obj_synth.deltaFF.max() > 1.2: ax2.set_ylim((-0.5,10))
            else:                            ax2.set_ylim((-0.1,1))

            # ------------------ spectrogams ----------------------------
            ax3 = fig.add_subplot(gs[2, 0])
            
            img = librosa.display.specshow(obj.Sxx_dB, x_axis="s", y_axis="linear", sr=obj.fs,
                         hop_length=obj.hop_length, ax=ax3, cmap=self.cmap)
            fig.colorbar(img, ax=ax3, format="%+2.f dB")
            #ax3.plot(obj.time, obj.FF, 'b*-', label='Real', ms=10)
            ax3.set_ylim(obj.flim); ax3.set_xlim((obj.time[0], obj.time[-1]))
            ax3.set_xticks(ax3.get_xticks(), ax3.get_xticklabels(), rotation=90, ha='right')
            ax3.yaxis.set_major_formatter(ticks)
            ax3.set_ylabel('f (khz)');  ax3.set_xlabel('');
            ax3.set_title('Real Spectrogram and FF')

            ax4 = fig.add_subplot(gs[3, 0])
            img = librosa.display.specshow(obj_synth.Sxx_dB, x_axis="s", y_axis="linear", sr=obj_synth.fs,
                         hop_length=obj_synth.hop_length, ax=ax4, cmap=self.cmap)
            fig.colorbar(img, ax=ax4, format="%+2.f dB")
            #ax4.plot(obj_synth.time, obj_synth.FF, 'go-', label='synthetic', ms=6)
            ax4.set_xlim((obj.time[0], obj.time[-1])); ax4.set_ylim(obj.flim);
            ax4.set_xticks(ax4.get_xticks(), ax4.get_xticklabels(), rotation=90, ha='right')
            ax4.yaxis.set_major_formatter(ticks)
            ax4.set_ylabel('f (khz)'); ax4.set_xlabel('time (s)');     
            ax4.set_title('Synthetic Spectrogram and FF')
            ax4.sharex(ax3)

            # ------------------ Mel spectgrograms ------------------
            ax5 = fig.add_subplot(gs[2, 2])
            img = librosa.display.specshow(obj.FF_coef, x_axis="s", y_axis="linear", sr=obj.fs,
                         hop_length=obj.hop_length, ax=ax5, cmap=self.cmap)
            fig.colorbar(img, ax=ax5, format="%+2.f dB")
            ax5.set_xlim((obj.time[0], obj.time[-1])); ax5.set_ylim(obj.flim); 
            ax5.set_xticks(ax5.get_xticks(), ax5.get_xticklabels(), rotation=90, ha='right')
            ax5.yaxis.set_major_formatter(ticks)
            ax5.set_ylabel('f (khz)'); ax5.set_xlabel('');
            ax5.set_title('Mel-Spectrogram Frequency Real (FF-R)')

            ax6 = fig.add_subplot(gs[3, 2])
            img = librosa.display.specshow(obj_synth.FF_coef, x_axis="s", y_axis="linear", sr=obj.fs,
                         hop_length=obj_synth.hop_length, ax=ax6, cmap=self.cmap)
            fig.colorbar(img, ax=ax6, format="%+2.f dB")
            ax6.set_xlim((obj.time[0], obj.time[-1])); ax6.set_ylim(obj.flim); 
            ax6.set_xticks(ax6.get_xticks(), ax6.get_xticklabels(), rotation=90, ha='right')
            ax6.yaxis.set_major_formatter(ticks)
            ax6.set_ylabel('f (khz)'); ax6.set_xlabel('time (s)');     
            ax6.set_title('Mel-Spectrogram Frequency Synth (FF-S)')
            ax6.sharex(ax5)

            # ------------------ Delta Sxx - Mell
            ax7 = fig.add_subplot(gs[2, 1])
            img = librosa.display.specshow(obj_synth.deltaSxx, x_axis="s", y_axis="linear", sr=obj.fs,
                         hop_length=obj_synth.hop_length, ax=ax7, cmap=self.cmap)
            fig.colorbar(img, ax=ax7, format="%+2.f dB")
            ax7.set_xticks(ax7.get_xticks(), ax7.get_xticklabels(), rotation=90, ha='right')
            ax7.yaxis.set_major_formatter(ticks)
            ax7.set_ylabel('f (khz) (s)'); ax7.set_xlabel('');
            ax7.set_ylim(obj.flim);  ax7.set_xlim((obj.time[0], obj.time[-1]));  
            ax7.set_title(r'Spectrum Error (ΔSxx), $||Sxx||_{}$={:.4f}'.format(obj_synth.ord, obj_synth.scoreMfccs ))#ax7.sharex(ax6)

            ax8 = fig.add_subplot(gs[3, 1])
            img = librosa.display.specshow(obj_synth.deltaMel, x_axis="s", y_axis="linear", sr=obj.fs,
                         hop_length=obj_synth.hop_length, ax=ax8, cmap=self.cmap)
            fig.colorbar(img, ax=ax8, format="%+2.f dB")
            ax8.set_xticks(ax8.get_xticks(), ax8.get_xticklabels(), rotation=90, ha='right')
            ax8.set_ylabel('f (khz) (s)'); ax8.set_xlabel('time (s)');  
            ax8.set_ylim(obj.flim);  ax8.set_xlim((obj.time[0], obj.time[-1])); 
            ax8.yaxis.set_major_formatter(ticks)
            ax8.set_title(r'Mel Normalized Error (ΔMel), $||Δmel||_{}$={:.4f}'.format(obj_synth.ord, obj_synth.scoreMfccs ))
            ax8.sharex(ax7)

            # ------------------ sound -------------------------
            ax9 = fig.add_subplot(gs[2, 3:4])
            ax9.plot(obj.time_s,       obj.s,             c='b', label='real')
            ax9.plot(obj.time_s,       obj.envelope,      c='k') #label='real_env',
            ax9.plot(obj_synth.time_s, 0.9*obj_synth.s,       c='g') #label='syn_env',
            ax9.plot(obj_synth.time_s, 0.9*obj_synth.envelope, c='g', label='synth')
            ax9.legend(); ax9.set_ylabel("Amplitud (a.u.)")
            ax9.set_title("Sound Waves")

            ax10 = fig.add_subplot(gs[3, 3:4])
            ax10.plot(obj_synth.time_s, obj_synth.deltaEnv, 'ko-', label=r' $||env||_{}$ = {:.4f},'.format(obj_synth.ord, obj_synth.scoreEnv)+'\nmean={:.4f}'.format(obj_synth.deltaEnv_mean))
            ax10.set_xlabel("t (s)"); ax10.set_ylabel("Amplitud (a.u.)"); 
            ax10.set_title("Envelope Difference (Δ env)"); 
            ax10.set_ylim((0,1)); ax10.legend()
            ax10.sharex(ax9)

            # ------------------ SIC -------------------------
            ax11 = fig.add_subplot(gs[2, 4])
            ax11.plot(obj.time, obj.SCI, 'go-', label='SCI real,   mean={:.2f}'.format(obj.SCI.mean()))
            ax11.plot(obj.time, obj_synth.SCI, 'bo-', label='SCI synth, mean={:.2f} '.format(obj_synth.SCI.mean()))

            ax11.set_xlabel("t (s)"); ax11.set_ylabel("SCI (adimensionless)"); 
            ax11.set_title("Spectral Content Index (SCI)"); 
            ax11.set_ylim((0,5)); ax11.legend()


            ax12 = fig.add_subplot(gs[3, 4])
            
            ax12.plot(obj.time, obj_synth.Df, 'H', label=r'$||DF||_{}$={:.3f},'.format(obj_synth.ord, obj_synth.scoreDF)+'\n  mean={:.3f} '.format(obj_synth.Df.mean()), ms=7)
            ax12.plot(obj.time, obj_synth.SKL, 's', color="purple", label=r'$||SKL||_{}$={:.3f},'.format(obj_synth.ord, obj_synth.scoreSKL)+'\n  mean={:.3f}'.format(obj_synth.SKL.mean()), ms=7)
            ax12.plot(obj.time, obj_synth.correlation, 'p', label=r'$||cor||_{}$={:.3f},'.format(obj_synth.ord, obj_synth.scoreCorrelation)+'\n   mean={:.3f} '.format(obj_synth.correlation.mean()))
            ax12.plot(obj_synth.time, obj_synth.deltaSCI, 'ko', label=r'$||SCI||_{}$={:.4f},'.format(obj_synth.ord, obj_synth.scoreSCI)+'\n   mean={:.3f}'.format(obj_synth.deltaSCI_mean))
            
            ax12.set_xlabel("t (s)"); ax12.set_ylabel("ΔSCI (adimensionless)"); 
            ax12.set_title("SCI Error and Acoustic Dissimilarity (ΔSCI & ADI)"); 
            if obj_synth.deltaSCI.max()>1.2: ax12.set_ylim((0,2)); 
            else:                            ax12.set_ylim((0,1)); 
            ax12.legend(bbox_to_anchor= (1.05, 1))


            fig.suptitle("SCORES", fontsize=20)
            plt.show()
            return fig, gs
            if self.save: fig.savefig(obj.paths.results+"ScoresVariables-{}-{}-{}.png".format(obj.id,obj.no_file,obj.no_syllable)) 

        else: print("Remember you must enter the object in the defined order: obj, obj_synth. \nEnter the objects again.")   
        
        
    def FindTimes(self, obj, FF_on=False):
        ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x*1e-3))
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))#, constrained_layout=True)
        
        img = librosa.display.specshow(obj.Sxx_dB, x_axis="s", y_axis="linear", sr=obj.fs,
                     hop_length=obj.hop_length, ax=ax, cmap=self.cmap)

        if FF_on:
            ax.plot(obj.time, obj.FF, "co", ms=8)#, label="Fundamental Frequency")
        
        # for i in range(len(obj.syllables)):  
        #     ax[0].plot([obj.time[obj.syllables[i][0]], obj.time[obj.syllables[i][-1]]], [0, 0], 'k', lw=5)
        #     ax[0].text((obj.time[obj.syllables[i][-1]]-obj.time[obj.syllables[i][0]])/2, 0.5, str(i))
        ax.set_ylim(obj.flim); ax.set_xlim(min(obj.time), max(obj.time));
        ax.set_title("Song Spectrum"); 
        ax.yaxis.set_major_formatter(ticks)
        ax.set_ylabel('f (kHz)');
        
        klicker = Klicker(fig, ax)
        
        return klicker