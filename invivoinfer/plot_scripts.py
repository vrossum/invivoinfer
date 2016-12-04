from __future__ import division
import matplotlib.pyplot as plt


def save_plot(filename,fig_name=None,fig_size=[2,2],font_size=6,file_format='eps'):
    if fig_name==None:
        fig_name=plt.gcf()
    
    fig_name.set_size_inches(fig_size)
    
    plt.savefig(filename,format=file_format)


