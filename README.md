# Variational-perspective-on-High-Resolution-ODES

In this project we provide a variational perspective on **high-resolution ODEs**. This will uncover the special choices of ODEs to recover accelerated algorithms and creates a framework to build Lyapunov functions. Our analysis lead to several improved convergence rates and stochastic algorithms. This work is currently Submitted to NIPS 2023.

## Code
```
git clone https://github.com/hoomyhh/Variational-perspective-on-High-Resolution-ODES.git
# python 3.11.3
```
### Training and Implementations
Here we train a simple CNN on CIFAR10 dataset. The aim is mainly the training error and behaviour of the **SGD,NNAG,SVRG, and NNAG+SVRG** algorithms.

```
python run_main.py --optimizer NNAG  --lr 0.05 --a 150
python run_main.py --optimizer SGD  --lr 0.01
python run_main.py --optimizer SVRG  --lr 0.01
python run_main.py --optimizer NNAG_SVRG  --lr 0.001 --a 30
```
### Output process
After running the algorithms and training, the .npz outputs are located in the /outputs directory. In case Monte_Carlo simulations were conducted **(for this you have to manually train as much as you need)** By manually, copy pasting the .npz files in separate folders for each algorithm (e.g. "path1","path2","path3", and "path4"), you can get the data to plot in Latex through this line of code
```
python load_data_all.py --path1 “path1” --path2  “path2” --path3 “path3” --"path4" --obj "objective" --outputfile "output_file_name"
```
The objective of the plot can be specified as
```
“train_error”
“train_acc”
“val_error”
“val_acc”
```
and the "output_file_name" is the optional name of the output file.

After the training we extract the data for plotting in Latex. The
default order of the final data file is SGD , SVRG ,  NNAG , NNAG+SVRG, Epoch , Lower1, Upper1,Lower2, Upper2,Lower3, Upper3,Lower4, Upper4

where the lower and upper refer to the lower and upper confidence intervals corresponding to each algorithm.
### Plot in Latex
```
\begin{figure}[!ht]
    \centering
    \begin{tikzpicture}
    
    \begin{groupplot}[group style={group size=2 by 1, horizontal sep=2cm},        
        width=0.475\textwidth, 
        height=0.395\textwidth,        
        xmin=0, xmax=100,        
        xlabel={Epoch},    
        % legend style={at={(0.5,1.3)}, anchor=north,legend columns=3},        
        % legend cell align={center},        
        grid=both, grid style={gray!30},        
        tick label style={font=\footnotesize}, 
        ]
    
    % First subplot: training error
    \nextgroupplot[
        ymin=0, ymax=2.5,
        ylabel={Training Error},
        ytick distance=0.5,
        line width=0.7pt,
        xtick align=outside,
        ytick align=outside,
        ]

    \addplot[red, fill=none, draw=none, name path=sgdlower] table[x=Epoch ,y=Lower1]{./NIPS 2023/Data/save_with_bounds_train_loss.txt};\label{SGD_train_plot_L}
    \addplot[red, fill=none, draw=none, name path=sgdupper] table[x=Epoch ,y=Upper1]{./NIPS 2023/Data/save_with_bounds_train_loss.txt};\label{SGD_train_plot_U}
    \addplot[red, opacity=0.075] fill between[of=sgdlower and sgdupper];
    \addplot[red, line width=1.25pt] table[x=Epoch ,y=SGD]{./NIPS 2023/Data/save_with_bounds_train_loss.txt};\label{SGD_train_plot}

    \addplot[orange, fill=none, draw=none, name path=svrglower] table[x=Epoch ,y=Lower2]{./NIPS 2023/Data/save_with_bounds_train_loss.txt};\label{SVRG_train_plot_L}
    \addplot[orange, fill=none, draw=none, name path=svrgupper] table[x=Epoch ,y=Upper2]{./NIPS 2023/Data/save_with_bounds_train_loss.txt};\label{SVRG_train_plot_U}
    \addplot[orange, opacity=0.075] fill between[of=svrglower and svrgupper];
    \addplot[orange, line width=1.25pt] table[x=Epoch ,y=SVRG]{./NIPS 2023/Data/save_with_bounds_train_loss.txt};\label{SVRG_train_plot}


    \addplot[blue, fill=none, draw=none, name path=nnaglower] table[x=Epoch ,y=Lower3]{./NIPS 2023/Data/save_with_bounds_train_loss.txt};\label{NNAG_train_plot_L}
    \addplot[blue, fill=none, draw=none, name path=nnagupper] table[x=Epoch ,y=Upper3]{./NIPS 2023/Data/save_with_bounds_train_loss.txt};\label{NNAG_train_plot_U}
    \addplot[blue, opacity=0.075] fill between[of=nnaglower and nnagupper];
    \addplot[blue, line width=1.25pt] table[x=Epoch ,y=NNAG]{./NIPS 2023/Data/save_with_bounds_train_loss.txt};\label{NNAG_train_plot}


    \addplot[cyan, fill=none, draw=none, name path=nnagsvrglower] table[x=Epoch ,y=Lower4]{./NIPS 2023/Data/save_with_bounds_train_loss.txt};\label{NNAG_SVRG_train_plot_L}
    \addplot[cyan, fill=none, draw=none, name path=nnagsvrgupper] table[x=Epoch ,y=Upper4]{./NIPS 2023/Data/save_with_bounds_train_loss.txt};\label{NNAG_SVRG_train_plot_U}
    \addplot[cyan, opacity=0.075] fill between[of=nnagsvrglower and nnagsvrgupper];
    \addplot[cyan, line width=1.25pt] table[x=Epoch ,y=NNAG+SVRG]{./NIPS 2023/Data/save_with_bounds_train_loss.txt};\label{NNAG_SVRG_train_plot}
    % \legend{SGD, SVRG, NNAG}
    
    % Second subplot: validation accuracy
    \nextgroupplot[
        ymin=0, ymax=0.85,    
        ylabel={Validation Accuracy},    
        ytick distance=0.1,
        line width=0.7pt,
        xtick align=outside,
        ytick align=outside,
        name = plot_val_acc,
        ]
        
    \addplot[red, fill=none, draw=none, name path=sgdlower] table[x=Epoch ,y=Lower1]{./NIPS 2023/Data/save_with_bounds_val_acc.txt};\label{SGD_val_plot_L}
    \addplot[red, fill=none, draw=none, name path=sgdupper] table[x=Epoch ,y=Upper1]{./NIPS 2023/Data/save_with_bounds_val_acc.txt};\label{SGD_val_plot_U}
    \addplot[red, opacity=0.1] fill between[of=sgdlower and sgdupper];
    \addplot[red, line width=1.5pt] table[x=Epoch ,y=SGD]{./NIPS 2023/Data/save_with_bounds_val_acc.txt};\label{SGD_val_plot}


    \addplot[orange, fill=none, draw=none, name path=svrglower] table[x=Epoch ,y=Lower2]{./NIPS 2023/Data/save_with_bounds_val_acc.txt};\label{SVRG_val_plot_L}               
    \addplot[orange, fill=none, draw=none, name path=svrgupper] table[x=Epoch ,y=Upper2]{./NIPS 2023/Data/save_with_bounds_val_acc.txt};\label{SVRG_val_plot_U}
    \addplot[orange, opacity=0.075] fill between[of=svrglower and svrgupper];
    \addplot[orange, line width=1.5pt] table[x=Epoch ,y=SVRG]{./NIPS 2023/Data/save_with_bounds_val_acc.txt};\label{SVRG_val_plot}           



    \addplot[blue, line width=1.5pt, fill=none, draw=none, name path=nnaglower] table[x=Epoch ,y=Lower3]{./NIPS 2023/Data/save_with_bounds_val_acc.txt};\label{NNAG_val_plot_L}
    \addplot[blue, line width=1.5pt, fill=none, draw=none, name path=nnagupper] table[x=Epoch ,y=Upper3]{./NIPS 2023/Data/save_with_bounds_val_acc.txt};\label{NNAG_val_plot_U}
    \addplot[blue, opacity=0.075] fill between[of=nnaglower and nnagupper];
    \addplot[blue, line width=1.5pt] table[x=Epoch ,y=NNAG]{./NIPS 2023/Data/save_with_bounds_val_acc.txt};\label{NNAG_val_plot}


    \addplot[cyan, line width=1.5pt, fill=none, draw=none, name path=nnagsvrglower] table[x=Epoch ,y=Lower4]{./NIPS 2023/Data/save_with_bounds_val_acc.txt};\label{NNAG_SVRG_val_plot_L}
    \addplot[cyan, line width=1.5pt, fill=none, draw=none, name path=nnagsvrgupper] table[x=Epoch ,y=Upper4]{./NIPS 2023/Data/save_with_bounds_val_acc.txt};\label{NNAG_SVRG_val_plot_U}
    \addplot[cyan, opacity=0.075] fill between[of=nnagsvrglower and nnagsvrgupper];
    \addplot[cyan, line width=1.5pt] table[x=Epoch ,y=NNAG+SVRG]{./NIPS 2023/Data/save_with_bounds_val_acc.txt};\label{NNAG_SVRG_val_plot}
    \end{groupplot}
    
    \node[anchor=south east, draw = black, line width=0.5pt, fill=white, font=\footnotesize]  (legend) at ([shift={(-0.1cm,0.1cm)}]plot_val_acc.south east) {\begin{tabular}{l l}
    SGD & \ref*{SGD_val_plot}  \\
    SVRG & \ref*{SVRG_val_plot} \\
    NNAG & \ref*{NNAG_val_plot} \\
    NNAG+SVRG & \ref*{NNAG_SVRG_val_plot}
    \end{tabular}};

    \end{tikzpicture}
```
