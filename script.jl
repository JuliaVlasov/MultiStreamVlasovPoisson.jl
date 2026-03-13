using JLD2
using JLD
using FileIO
using CSV
using LaTeXStrings
using Plots
using FFTW
using LaTeXStrings

str_ext=".png"

#load the files to transform into Julia arrays
##############################################


#Time dependent energies 
########################
toto1  = CSV.read("frame1.csv",DataFrame);
toto2  = CSV.read("frame2.csv",DataFrame);

s=size(toto1);Nt=s[1];tf=toto1[Nt,1]
t=toto1[!,:time];
tmp1=toto1[!,:elec];
str="elec_energy_main1"
plot(toto1.time,toto1.elec,lw=4,xlabel=L"\omega_p t",labelfontsize=26,xtickfont=font(14),ytickfont=font(14),legend=false,ylabel=L"\log\;|\!\!|E_x|\!\!|",framestyle = :box,ylims=(-5,-1))
#plot(time,elec,xlabel="time",ylabel=str,legend=false)
filename=string(str)
savefig(filename)



#str=string("log_Ex_energy_zoom", str_file1)
#plot(frame1.Time,log.(sqrt.(frame1.PotentialEnergyE1)),lw=4,xlabel=L"\omega_p t",labelfontsize=26,xtickfont=font(14),ytickfont=font(14),legend=false,ylabel=L"\#log\;|\!\!|E_x|\!\!|",framestyle = :box,ylims=(-5,-1))
#plot!(frame1.Time[1:5000],log.(sqrt.(frame1.PotentialEnergyE1[1:5000])),legend=false,framestyle=:box, subplot = 2,inset=(1,bbox(0.,0.,0.6,0.5, :top, :right)),l#w=2,ylims=(-5,-1),title=L"v_{th}=0.17",titlefontsize=32,guidefont=font(22),top_margin=6mm)
#filename=string(str)
#savefig(filename)
