from .functions import *

class Ploter(object): 
    def __init__(self, save=False):
        self.save = save
        
    # ------------- bird ----------
    def PlotSong(self,obj, syllable_on=False, chunck_on=False): 
        fig, ax = plt.subplots(2+int(syllable_on), 1, figsize=(12, 6+3*int(syllable_on)))
        fig.subplots_adjust(hspace=0.4, wspace=0.4)


        ax[0].pcolormesh(obj.tu, obj.fu/1000, obj.Sxx, cmap=plt.get_cmap('Greys'), rasterized=True)
        for i in range(len(obj.syllables)):  
            ax[0].plot([obj.time[obj.syllables[i][0]], obj.time[obj.syllables[i][-1]]], [0, 0], 'k', lw=5)
            ax[0].text((obj.time[obj.syllables[i][-1]]-obj.time[obj.syllables[i][0]])/2, 0.5, str(i))
        ax[0].set_ylim(0, 12.000); ax[0].set_xlim(min(obj.time), max(obj.time));
        ax[0].set_title("Complete Song Spectrum"); 
        ax[0].set_ylabel('f (kHz)'); #ax[0].set_xlabel('t (s)'); 
        

        ax[1].plot(obj.time, obj.s/np.max(obj.s),'k', label='audio')
        ax[1].plot(obj.time, np.ones(len(obj.time))*obj.umbral, '--', label='umbral')
        ax[1].plot(obj.time, obj.envelope, label='envelope')
        ax[1].legend(loc='upper right')#loc=1, title='Data')
        ax[1].set_title("Complete Song Sound Wave")
        ax[1].set_xlabel('t (s)'); ax[1].set_ylabel('Amplitud normalaized');
        ax[1].sharex(ax[0])

        if chunck_on and syllable_on:
            ax[0].plot(obj.chunck.timeFF+obj.chunck.t0, obj.chunck.FF*1e-3, 'g-', label='Chunck', lw=10)
            ax[2].plot(obj.chunck.timeFF+obj.chunck.t0, obj.chunck.FF*1e-3, 'g-', label='Chunck', lw=8)

        if syllable_on:
            ax[0].plot(obj.syllable.timeFF+obj.syllable.t0, obj.syllable.FF*1e-3, 'bo', label='Syllable'.format(obj.syllable.fs), lw=1)

            ax[2].pcolormesh(obj.syllable.tu+obj.syllable.t0, obj.syllable.fu*1e-3, obj.syllable.Sxx, cmap=plt.get_cmap('Greys'), rasterized=True) 
            ax[2].plot(obj.syllable.timeFF+obj.syllable.t0, obj.syllable.FF*1e-3, 'b+', label='Syllable', ms=10)
            ax[2].set_ylim((2, 11)); 
            ax[2].set_xlim((obj.syllable.timeFF[0]-0.001+obj.syllable.t0, obj.syllable.timeFF[-1]+0.001+obj.syllable.t0));
            ax[2].legend(loc='upper right', title="FF")
            ax[2].set_xlabel('t (s)'); ax[2].set_ylabel('f (khz)')
            ax[2].set_title('Single Syllable Spectrum, No {}'.format(obj.no_syllable))

            ax[0].legend(loc='upper right', title="FF")
            
            path_save = obj.paths.results+"AllSongAndSyllable-{}-{}.png".format(obj.no_file,obj.no_syllable)
        else: path_save = obj.paths.results+"AllSongAndSyllable-{}.png".format(obj.no_file)
                
        fig.suptitle('Audio: {}'.format(obj.file_name[39:]), fontsize=18)
        plt.show()

        if self.save: fig.savefig(path_save)

    def PlotSynth(self, obj):

        fig, ax = plt.subplots(2, 2, figsize=(12, 5), sharex=True, sharey='col')
        fig.subplots_adjust(top=0.85)     

        ax[0][0].plot(obj.time, obj.s, label='canto', c='b')
        ax[0][0].set_title('Real')
        ax[0][0].plot(obj.time, obj.envelope, label='envelope', c='k')
        ax[0][0].legend(); ax[0][0].set_ylabel("Amplitud (a.u.)")
        ax[1][0].plot(obj_synth.time, obj_synth.s, label='synthetic', c='g')
        ax[1][0].set_title('Synthetic') 
        ax[1][0].plot(obj_synth.time, obj_synth.envelope
    , label='envelope', c='k')
        ax[1][0].legend(); ax[1][0].set_xlabel('t (s)'); ax[1][0].set_ylabel("Amplitud (a.u.)")

        Delta_tu   = obj.tu[-1] - obj.tu[0]
        Delta_tu_s = 1#tu_s[-1] - tu_s[0]

        pcm = ax[0][1].pcolormesh(obj.tu, obj.fu*1e-3, obj.Sxx_dB, cmap=plt.get_cmap('Greys'), rasterized=True)#, vmin=10, vmax=70)
        fig.colorbar(pcm, ax=ax[0,1], location='right', label='Power (dB)')
        ax[0][1].plot(obj.timeFF, obj.FF*1e-3, 'bo-', lw=2)
        ax[0][1].set_title('Real'); ax[0][1].set_ylabel('f (khz)'); ax[0][1].set_ylim(0, 20);

        pcm = ax[1][1].pcolormesh(obj_synth.tu, obj_synth.fu*1e-3, obj_synth.Sxx_dB, cmap=plt.get_cmap('Greys'), rasterized=True)#, vmin=10, vmax=70)
        fig.colorbar(pcm, ax=ax[1,1], location='right', label='Power (dB)')
        ax[1][1].plot(obj_synth.timeFF, obj_synth.FF*1e-3, 'go-', lw=2)
        ax[1][1].set_title('Synthetic') 
        ax[1][1].set_ylim(0, 20);   ax[1][1].set_xlim(min(obj_synth.time), max(obj_synth.time))
        ax[1][1].set_xlabel('t (s)'); ax[1][1].set_ylabel('f (khz)');

        #fig.tight_layout(); 
        fig.suptitle('Sound Waves and Spectrograms', fontsize=20)
        plt.show()

        if self.save: fig.savefig(obj.paths.results+"SoundAndSpectros-{}-{}-{}.png".format(obj.id,obj.no_file,obj.no_syllable))

    def PlotAlphaBeta(self, obj, xlim=(-0.05,.2), ylim=(-0.2,0.9)):
        fig = plt.figure(constrained_layout=True, figsize=(10, 6))
        gs  = GridSpec(2, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1:, 0])
        ax3 = fig.add_subplot(gs[:, 1])

        fig.suptitle("GridSpec")
        fig.tight_layout(pad=3.0) #fig.subplots_adjust(top=0.85)

        viridis = cm.get_cmap('Blues')
        c       = viridis(np.linspace(0.3, 1, np.size(obj.time)))

        ax1.scatter(obj.time, obj.alpha, c=c, label="alfa")
        ax1.set_title('Air-Sac Pressure')
        ax1.grid()
        ax1.set_ylabel('α (a.u.)'); #ax1.set_xlabel('Time (s)'); 
        ax1.set_ylim(xlim);        #ax[0].legend()

        ax2.scatter(obj.time, obj.beta, c=c, label="beta")
        ax2.set_title('Labial Tension')
        ax2.grid()
        ax2.set_xlabel(r'Time (s)'); ax2.set_ylabel('β (a.u.)')
        ax2.set_ylim(ylim);         ax2.sharex(ax1);        #ax[0].legend()

        # ------------- Bogdanov–Takens bifurcation ------------------
        mu2 = np.linspace(-2.5, 1/3, 1000)
        xmas = (1+np.sqrt(1-3*mu2))/3
        xmen = (1-np.sqrt(1-3*mu2))/3
        mu1_mas = -mu2*xmas - xmas**3 + xmas**2
        mu1_men = -mu2*xmen - xmen**3 + xmen**2
        #---------------------------------------------------------------

        ax3.scatter(obj.alpha, obj.beta, c=c, label="Parameters", marker="_")
        ax3.plot(-1/27, 1/3, 'ko')#, label="Cuspid Point"); 
        ax3.axvline(0, color='red', lw=1)#, label="Hopf Bifurcation")
        ax3.plot(mu1_men, mu2, '-g', lw=1)#, label="Saddle-Noddle\nBifurcation"); 
        ax3.text(-0.01,0.6, "Hopf",rotation=90, color="r")
        ax3.text(-0.04,0.39,"CP",  rotation=0,  color="k")
        ax3.text(-0.03,0.2,  "SN",  rotation=0,  color="g")
        ax3.text(0.1, 0.005, "SN",  rotation=0,  color="g")

        ax3.plot(mu1_mas, mu2, '-g', lw=1)
        ax3.fill_between(mu1_mas, mu2, 10, where=mu1_mas > 0, color='gray', alpha=0.25)#, label='BirdSongs')
        ax3.set_ylabel('Tension (a.u.)'); ax3.set_xlabel('Pressure (a.u.)')
        ax3.set_title('Parameter Space')
        ax3.legend()
        ax3.set_xlim(xlim); ax3.set_ylim(ylim)
        fig.suptitle("Air-Sac Pressure (α) and Labial Tension (β) Parameters", fontsize=20)#, family='fantasy')
        plt.show()

        if self.save: fig.savefig(obj.paths.results+"MotorGesturesParameters-{}-{}-{}.png".format(obj.id,obj.no_file,obj.no_syllable))


    def Plot(self, obj, obj_synth, cmp="afmhot_r"):       
        fig = plt.figure(constrained_layout=False, figsize=(28, 12))
        gs  = fig.add_gridspec(nrows=4, ncols=5, wspace=0.05, hspace=0.2)
        vmin, vmax = obj.Sxx_dB.min(), obj.Sxx_dB.max()
        # ----- FF ---------------
        ax1 = fig.add_subplot(gs[0:2, :3])
        pcm = ax1.pcolormesh(obj.tu, obj.fu*1e-3, obj.Sxx_dB, cmap=plt.get_cmap(cmp), rasterized=True, vmin=vmin, vmax=vmax)
        plt.colorbar(pcm, ax=ax1, location='left', label='Power (dB)', pad=0.025)
        
        ax1.plot(obj.FF_time,       obj.f_msf*1e-3,       'D-', color="skyblue", label=r'$FF_{msf}$ real',ms=12)
        ax1.plot(obj_synth.FF_time, obj_synth.f_msf*1e-3, 'X-', color="lightgreen",label=r'$FF_{msf}$ synt', ms=12)
        
        ax1.plot(obj.timeFF,        obj.FF*1e-3,       'b*-', label=r'FF real',ms=25)
        ax1.plot(obj_synth.timeFF,  obj_synth.FF*1e-3, 'go-', label=r'FF synt', ms=12)
        
        ax1.plot(obj_synth.FF_time, obj_synth.centroid*1e-3, 'p-', color="olive",  label=r'$F_{cent}$ synth', ms=12)
        ax1.plot(obj.FF_time,       obj.centroid*1e-3,       '+-', color="yellow", label=r'$F_{cent}$ real', ms=12)
        
        ax1.legend(borderpad=0.6, labelspacing=0.7, title="Feature"); ax1.set_ylim((1, 20)); 
        ax1.set_xlim((obj.tu[0], obj.tu[-1]))
        ax1.set_ylabel('f (khz)'); ax1.set_xlabel('time (s)');     
        ax1.set_title('Spectrogram - Fundamental Frequency (FF)')

        ax2 = fig.add_subplot(gs[0:2, 3:])
        
        ax2.plot(obj_synth.timeFF,  obj_synth.deltaFF ,      "*-", color="k",  ms=12, lw=3, label=r' $||ΔFF||_{}$= {:.4f}, mean={:.4f}'.format(obj.ord, obj_synth.scoreFF, obj_synth.deltaFF_mean)); 
        ax2.plot(obj_synth.FF_time, obj_synth.deltaRMS,      "-p", color="r",  label=r' $|| ΔF_{{ rms }}||_{}$= {:.4f}, mean={:.4f}'.format(obj.ord,  obj_synth.scoreRMS, obj_synth.scoreRMS_mean)); 
        ax2.plot(obj.FF_time,       obj_synth.deltaCentroid,  "-o", color="y", label=r'$ || \Delta F_{{ centroid }}||_{}$ = {:.4f}, mean={:.4f}'.format(obj.ord, obj_synth.scoreCentroid, obj_synth.scoreCentroid_mean)); 
        ax2.plot(obj.FF_time,       obj_synth.deltaF_msf,     "D-", color="purple", label=r'$ || \Delta F_{{ msf }}||_{}$ = {:.4f}, mean={:.4f}'.format(obj.ord, obj_synth.scoreF_msf, obj_synth.scoreF_msf_mean)); 
        
        
        ax2.plot(obj_synth.FF_time, obj_synth.rms*1e-3, 'p-', color="darkred", label=r'$F_{rms}$ synth', ms=12)
        ax2.plot(obj.FF_time,       obj.rms*1e-3,       'rv-', label=r'$F_{rms}$ real', ms=12)
        
        ax2.set_xlabel('time (s)'); ax2.set_ylabel('f (kHz)'); ax2.legend(title="Features")
        ax2.set_title('Fundamental Frequency Error (ΔFF)'); 
        if obj_synth.deltaFF.max() > 1.2: ax2.set_ylim((-0.5,10))
        else:                            ax2.set_ylim((-0.1,1))


        # ------------------ spectgrograms
        ax3 = fig.add_subplot(gs[2, 0])
        pcm = ax3.pcolormesh(obj.tu, obj.fu*1e-3, obj.Sxx_dB, cmap=plt.get_cmap(cmp), rasterized=True, vmin=vmin, vmax=vmax)
        plt.colorbar(pcm, ax=ax3, location='left', label='Power (dB)', pad=0.05)
        ax3.plot(obj.timeFF, obj.FF*1e-3, 'b*-', label='Real', ms=10)
        ax3.set_ylim((1, 15)); ax3.set_xlim((obj.tu[0], obj.tu[-1]))
        ax3.set_ylabel('f (khz)');    
        ax3.set_title('Real Spectrogram and FF)')

        ax4 = fig.add_subplot(gs[3, 0])
        pcm = ax4.pcolormesh(obj_synth.tu, obj_synth.fu*1e-3, obj_synth.Sxx_dB, cmap=plt.get_cmap(cmp), vmin=vmin, vmax=vmax)
        plt.colorbar(pcm, ax=ax4, location='left', label='Power (dB)', pad=0.05)
        ax4.plot(obj_synth.timeFF, obj_synth.FF*1e-3, 'go-', label='synthetic', ms=6)
        ax4.set_xlim((obj.tu[0], obj.tu[-1])); ax4.set_ylim((1, 15)); 
        ax4.set_ylabel('f (khz)'); ax4.set_xlabel('time (s)');     
        ax4.set_title('Synthetic Spectrogram and FF')
        ax4.sharex(ax3)

        # ------------------ Mel spectgrograms
        ax5 = fig.add_subplot(gs[2, 2])
        pcm = ax5.pcolormesh(obj.FF_time, obj.freq*1e-3, obj.FF_coef, rasterized=True, cmap=plt.get_cmap(cmp))#, vmin=10, vmax=70)
        plt.colorbar(pcm, ax=ax5, location='left', label='Power', pad=-0.05)
        ax5.set_xlim((obj.tu[0], obj.tu[-1])); ax5.set_ylim((1, 15)); 
        ax5.set_ylabel('f (khz)'); #ax5.set_xlabel('time (s)');     
        ax5.set_title('Mel-Spectrogram Frequency Real (FF-R)')
        #ax5.sharex(ax3)

        ax6 = fig.add_subplot(gs[3, 2])
        pcm = ax6.pcolormesh(obj_synth.FF_time, obj_synth.freq*1e-3, obj_synth.FF_coef, rasterized=True, cmap=plt.get_cmap(cmp))#, vmin=10, vmax=70)
        fig.colorbar(pcm, ax=ax6, location='left', label='Power', pad=-0.05)
        ax6.set_xlim((obj.tu[0], obj.tu[-1])); ax5.set_ylim((1, 15)); 
        ax6.set_ylabel('f (khz)'); ax6.set_xlabel('time (s)');     
        ax6.set_title('Mel-Spectrogram Frequency Synth (FF-S)')
        ax6.sharex(ax5)

        # ------------------ Delta Sxx - Mell
        ax7 = fig.add_subplot(gs[2, 1])
        pcm = ax7.pcolormesh(obj_synth.tu, obj.fu*1e-3, obj_synth.deltaSxx, cmap=plt.get_cmap(cmp), rasterized=True)#, vmin=0, vmax=1)
        plt.colorbar(pcm, ax=ax7, location='left', label='Power (adimensionless)', pad=-0.05)
        #ax[1][0].plot(obj_synth.timeFF, obj_synth.deltaSCI, "-o", color="k", label='Σ R(SCI) = {:.4f}'.format(obj_synth.scoreSCI))
        ax7.set_ylabel('f (khz) (s)'); ax7.set_xlabel(''); ax7.set_ylim((1, 15)); 
        ax7.set_title(r' Spectrum Error (ΔSxx), $||Sxx||_{}$={:.4f}'.format(obj.ord, obj_synth.scoreMfccs ))#ax7.sharex(ax6)

        ax8 = fig.add_subplot(gs[3, 1])
        pcm = ax8.pcolormesh(obj_synth.FF_time, obj_synth.freq*1e-3, obj_synth.deltaMel,  rasterized=True, cmap=plt.get_cmap(cmp))#,, vmin=0, vmax=1)
        plt.colorbar(pcm, ax=ax8, location='left', label='Power (adimensionless)', pad=-0.05)
        ax8.set_ylabel('f (khz) (s)'); ax8.set_xlabel('t (s)');  ax8.set_ylim((1, 15)); 
        ax8.set_title(r'Mel Normalized Error (ΔMel), $||Δmel||_{}$={:.4f}'.format(obj.ord, obj_synth.scoreMfccs ))

        # ------------------ sound
        ax9 = fig.add_subplot(gs[2, 3])
        ax9.plot(obj.time,       obj.s,             c='b', label='real')
        ax9.plot(obj.time,       obj.envelope,      c='k') #label='real_env',
        ax9.plot(obj_synth.time, obj_synth.s,       c='g') #label='syn_env',
        ax9.plot(obj_synth.time, obj_synth.envelope, c='g', label='synth')
        ax9.legend(); ax9.set_ylabel("Amplitud (a.u.)")
        ax9.set_title("Sound Waves")

        ax10 = fig.add_subplot(gs[3, 3])
        ax10.plot(obj_synth.time, obj_synth.deltaEnv, 'ko-', label=r' $||env||_{}$ = {:.4f}, mean={:.4f}'.format(obj.ord, obj_synth.scoreEnv, obj_synth.deltaEnv_mean))
        ax10.set_xlabel("t (s)"); ax10.set_ylabel("Amplitud (a.u.)"); 
        ax10.set_title("Envelope Difference (Δ env)"); 
        ax10.set_ylim((0,1)); ax10.legend()
        ax10.sharex(ax9)


        # ------------------ SIC
        ax11 = fig.add_subplot(gs[2, 4])
        ax11.plot(obj.FF_time, obj.SCI, 'go-', label='SCI real,   mean={:.2f}'.format(obj.SCI.mean()))
        ax11.plot(obj.FF_time, obj_synth.SCI, 'bo-', label='SCI synth, mean={:.2f} '.format(obj_synth.SCI.mean()))
        
        
        ax11.set_xlabel("t (s)"); ax11.set_ylabel("SCI (adimensionless)"); 
        ax11.set_title("Spectral Content Index (SCI)"); 
        ax11.set_ylim((0,5)); ax11.legend()



        ax12 = fig.add_subplot(gs[3, 4])
        ax12.plot(obj_synth.FF_time, obj_synth.deltaSCI, 'ko-', label=r'$||SCI||_{}$={:.4f}, mean={:.3f}'.format(obj.ord, obj_synth.scoreSCI, obj_synth.deltaSCI_mean))
        
        ax12.plot(obj.FF_time, obj_synth.correlation, 'p-', label=r'$||cor||_{}$={:.3f}, mean={:.3f} '.format(obj.ord, obj_synth.scoreCorrelation, obj_synth.correlation.mean()))
        ax12.plot(obj.FF_time, obj_synth.Df, 'H-', label=r'$||DF||_{}$={:.3f}, mean={:.3f} '.format(obj.ord, obj_synth.scoreDF, obj_synth.Df.mean()))
        ax12.plot(obj.FF_time, obj_synth.SKL, 's-', color="tomato", label=r'$||SKL||_{}$={:.3f}, mean={:.3f} '.format(obj.ord, obj_synth.scoreSKL, obj_synth.SKL.mean()))
        
        ax12.set_xlabel("t (s)"); ax12.set_ylabel("ΔSCI (adimensionless)"); 
        ax12.set_title("SCI Error and Acoustic Dissimilarity (ΔSCI & ADI)"); 
        if obj_synth.deltaSCI.max()>1.2: ax12.set_ylim((0,2)); 
        else:                            ax12.set_ylim((0,1)); 
        ax12.legend()


        fig.suptitle("SCORES", fontsize=20)
        plt.show()

        if self.save: fig.savefig(obj.paths.results+"ScoresVariables-{}-{}-{}.png".format(obj.id,obj.no_file,obj.no_syllable)) 


    def PlotVs(self,obj, xlim=(0,0.025)):
        fig, ax = plt.subplots(3, 1, figsize=(12, 9))
        fig.subplots_adjust(wspace=0.35, hspace=0.4)

        time = obj.time[:obj.Vs.shape[0]]

        ax[0].plot(obj.Vs[:,4], color='b')
        #ax[0].set_xlim((0,1e5))
        ax[0].set_xlabel("time (s)"); ax[0].set_ylabel("$P_{out}$");
        ax[0].set_title("Trachea Output Pressure")

        ax[1].plot(obj.Vs[:,1], color='g') 
        #ax[1].set_xlim(xlim)
        ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("$P_{in}$");
        ax[1].set_title("Trachea Input Pressure")

        ax[2].plot(obj.Vs[:,0], color='r')
        #ax[2].set_xlim(xlim)
        ax[2].set_xlabel("time (s)"); ax[2].set_ylabel("$x(t)$");
        ax[2].set_title("Labial position")
        ax[2].sharex(ax[1])

        fig.suptitle('Labial Parameters (vector $v$)', fontsize=20)
        plt.show()

        if self.save: fig.savefig(obj.paths.results+"MotorGesturesVariables-{}-{}-{}.png".format(obj.id,obj.no_file,obj.no_syllable)) 