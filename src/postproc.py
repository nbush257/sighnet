from brian2 import *
import os
import scipy.io.matlab as sio
import scipy.signal
import pickle
import glob
import click
import pandas as pd


def smooth_rate(ratemon, width,mode='gaussian'):
    dt = np.mean(np.diff(ratemon['t']))
    if mode == 'gaussian':
        width_dt = int(np.round(2 * width / dt))
        window = np.exp(-np.arange(-width_dt,
                                   width_dt + 1) ** 2 *
                        1. / (2 * (width / dt) ** 2))
    elif mode == 'flat':
        width_dt = int(width / 2 / dt) * 2 + 1
        window = np.ones(width_dt)
    else:
        raise ValueError
    return (Quantity(np.convolve(ratemon['rate'],
                                 window * 1. / sum(window),
                                 mode='same'), dim=hertz.dim))


def post_proc_ko_sweep(flist,fout,savemat=True,savecsv=True):
    '''
    Get the smoothed population rates, sigh, and eupnic burst rates from a list of
    dat files. Allows to plot population rates and sigh frequency as a function of
    Ko
    :param flist: A list of outputs from "sigh_net.py". Must be full path names
    :param fout: filename to save to
    :param savemat: Pass True to save the .mat file of all processed data (default:True)
    :param savecsv: Pass True to save the .csv file of sigh and eupnic frequencies (default:True)
    :return: Saves a .mat file with the following data:
                - smoothed_rates    : smoothed population spike rates for a subset of the data (100 seconds)
                - dt                : timestep
                - sigh_shapes       : dictionary of sigh burst arrays (time x N_sighs)
                - eup_shapes        : dictionary of eupnea burst arrays (time x N_sighs)
                - t_sigh            : time window vector to plot sigh shapes
                - t_eup             : time window vector to plot eup shapes
                - ISI               : array of intersigh intervals in the order of "ko_list"
                - IEI               : array of intereupnea intervals in the order of "ko_list"
                - ko_list           : list of ko values
    '''

    if not savemat and not savecsv:
        print('No save options chosen')
        return(-1)

    flist.sort()

    # Init lists
    ISI = [] # intersigh interval
    IEI = [] # intereupnea interval
    ko_list = []
    Rs_all = [] # smoothed rates
    all_sigh_shape = []
    all_eup_shape = []


    # Loop through each file to append the extracted data
    # to lists
    for dat_fn in flist:
        print(dat_fn)
        prefix = os.path.split(os.path.splitext(dat_fn)[0])[1]
        # Load in the data
        with open(dat_fn,'rb') as fid:
            dat = pickle.load(fid)

        # Pull out the Ko value from metadata
        ko_val = np.median(dat['params']['ko'])
        ko_list.append(ko_val)

        # Get the smoothed population rate
        dt_r = dat['pop_rate']['t'][1] - dat['pop_rate']['t'][0]
        R = dat['pop_rate']
        Rs = smooth_rate(R, 50 * ms,mode='flat')
        Rs_all.append(Rs)

        # Store the peak times
        pk,props = scipy.signal.find_peaks(Rs,prominence=300,height=300,distance=1000)
        wd,_,left_onset,_ = scipy.signal.peak_widths(Rs,pk,rel_height=0.8)
        next_IBI = np.hstack([0,np.diff(pk)])
        next_IBIz = scipy.stats.zscore(next_IBI)

        # Decompose peaks by normalized amplitude (ht) and duration (wd)
        ht = props['peak_heights']
        htz = scipy.stats.zscore(ht)
        wdz = scipy.stats.zscore(wd)

        ## Seperate sighs from eupneas
        # At values of Ko above ~5 with parameters used in publication,
        # there are no longer enough eupneas to make sighs outliers
        if ko_val <4.8:
            idx_sigh = np.logical_and(htz>1,wdz>1,next_IBIz>1)
            idx_eup = np.logical_not(idx_sigh)
        else:
            idx_sigh = ht>425
            idx_eup = ht<=425
        pk_eup = left_onset[idx_eup].astype('int')
        pk_sigh = left_onset[idx_sigh].astype('int')
        ISI.append(np.mean(np.diff(pk_sigh))*dt_r)
        IEI.append(np.mean(np.diff(pk_eup))*dt_r)

        # Store the sigh shapes
        pre_win = int(0.5*second /dt_r)
        post_win = int(2*second/dt_r)
        sigh_shape = np.empty([pre_win+post_win,len(pk_sigh)])
        for ii,pk in enumerate(pk_sigh):
            if (pk-pre_win)<0:
                continue
            sigh_shape[:,ii] = Rs[pk-pre_win:pk+post_win]
        all_sigh_shape.append(sigh_shape)

        # Store the eup shapes
        pre_win = int(0.5*second /dt_r)
        post_win = int(1*second/dt_r)
        eup_shape = np.empty([pre_win+post_win,len(pk_eup)]) *np.nan
        for ii,pk in enumerate(pk_eup[1:-1]):
            if pk-pre_win<0:
                continue
            eup_shape[:,ii] = Rs[pk-pre_win:pk+post_win]
        eup_shape  = np.nanmedian(eup_shape,axis=1)
        all_eup_shape.append(eup_shape)

    # Order all lists by ascending ko
    ko_order = np.argsort(ko_list)
    ko_list = np.sort(ko_list)
    Rs_all = [ x for _, x in sorted(zip(ko_order, Rs_all))]
    all_sigh_shape = [x for _, x in sorted(zip(ko_order, all_sigh_shape))]
    all_eup_shape = [ x for _, x in sorted(zip(ko_order, all_eup_shape))]
    ISI = [ x for _, x in sorted(zip(ko_order, ISI))]
    IEI = [ x for _, x in sorted(zip(ko_order, IEI))]


    Rs_all_array = np.array(Rs_all)[:,int(100*second/dt_r):int(200*second/dt_r)]
    sigh_shape_dict = {k: v for k, v in zip(ko_list, all_sigh_shape)}
    eup_shape_dict = {k: v for k, v in zip(ko_list, all_eup_shape)}
    t_sigh = np.arange(0 * second, sigh_shape.shape[0] * dt_r, dt_r) - 0.5 * second
    t_eup = np.arange(0 * second, eup_shape.shape[0] * dt_r, dt_r) - 0.5 * second
    if savemat:
        print('Saving mat file')
        save_dict = {
                 'smooth_rates':Rs_all_array,
                 'dt':dt_r,
                 'sigh_shapes':sigh_shape_dict,
                 'eup_shapes':eup_shape_dict,
                 't_sigh':t_sigh,
                 't_eup':t_eup,
                 'ISI':ISI,
                 'IEI':IEI,
                 'ko_list':ko_list}
        outfile = f'./{fout}_proc.mat'
        sio.savemat(outfile,save_dict)
        print(f'Saved processed data to {outfile}')
    if savecsv:
        df = pd.DataFrame()
        df['ko'] = ko_list
        df['intersigh_interval'] = ISI
        df['intereupnic_interval'] = IEI

        outfile = f'./{fout}_ISI.csv'
        df.to_csv(outfile,index=False)
        print(f'Saved burst intervals to {outfile}')

    return(0)


@click.command()
@click.argument('fspec')
@click.option('--savemat',is_flag=True)
@click.option('--savecsv',is_flag=True)
def main(fspec,savemat,savecsv):
    '''
    Postprocesses all files that match the strin specifier\n
    ex:\n
    directory contains: sighnet_test_ko00.1, sighnet_test_ko01.0.dat, sighnet_test_ko02.0.dat\n
    \trun:\n
    \tpython postproc.py "./sighnet_test*.dat" --savecsv\n
    '''

    flist = glob.glob(fspec)
    fout = os.path.splitext(fspec.replace('*',''))[0]
    post_proc_ko_sweep(flist,fout,savemat,savecsv)
    return(0)

if __name__ == '__main__':
    main()


