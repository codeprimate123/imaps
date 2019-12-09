"""Comparative visualisation of specific motifs across data or regions

The user should define:
1. List of 1 or more kmers that will be visualized as a group (based on average counts)
2. List of data (1 or more) to compare. Data is in form of bed files containing crosslinks.
3. List of  genomic regions (1 or more) to compare. possible regions are 'genome', 'intergenic', 
'intron', 'UTR3', 'other_exon'(comprised of 'UTR5' and 'CDS') and 'ncRNA'.

First step is regional thresholding to obtain thresholded crosslinks. This approach takes crosslinks
in all peaks within a region to define threshold and so introduces an element of intra-regional comparison.
Regions for thresholding as defined in the following way:
- all exons in the same gene (5’UTR, CDS, 3’UTR, or all exons in ncRNAs) are considered one region,
- each intron is its own region,
- each intergenic region is its own region.

Draw the selected kmer positional distribution around thresholded xcrosslinks (-150..100) for the 
multiple data/regions as defined by a user. If multiple data AND regions are given, 
then each region is shown on a separate plot.
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pybedtools as pbt
from random import randint
from plumbum import local
from itertools import islice


REGION_SITES = {
    'genome': ['intron', 'CDS', 'UTR3', 'UTR5', 'ncRNA', 'intergenic'],
    'whole_gene': ['intron', 'CDS', 'UTR3', 'UTR5'],
    'intergenic': ['intergenic'],
    'intron': ['intron'],
    'ncRNA': ['ncRNA'],
    'other_exon': ['UTR5', 'CDS'],
    'UTR3': ['UTR3'],
    'UTR5': ['UTR5']
}
REGIONS_QUANTILE = [
    'intron',
    'intergenic',
    'cds_utr_ncrna',
]
REGIONS_MAP = {}
TEMP_PATH = None

# overriding pybedtools to_dataframe method to avoid warning
def to_dataframe_fixed(self, *args, **kwargs):
    """
    Create a pandas.DataFrame, passing args and kwargs to pandas.read_csv.

    This function overrides pybedtools function to avoid FutureWarning:
    read_table is deprecated, use read_csv instead... Pandas must be
    imported as pd, it is advisable to specify dtype and names as well.
    """
    return pd.read_csv(self.fn, header=None, sep='\t', *args, **kwargs)


pbt.BedTool.to_dataframe = to_dataframe_fixed  # required for overriding


def get_name(s_file):
    """Return sample name from file path."""
    return s_file.split('/')[-1].replace('.gz', '').replace('.bed', "").replace('.xl', "")


def parse_bed6_to_df(p_file):
    """Parse BED6 file to pandas.DataFrame."""
    return pd.read_csv(
        p_file,
        names=['chrom', 'start', 'end', 'name', 'score', 'strand'],
        sep='\t',
        header=None,
        dtype={'chrom': str, 'start': int, 'end': int, 'name': str, 'score': float, 'strand': str}
    )


def parse_region_to_df(region_file):
    """Parse GTF to pandas.DataFrame."""
    return pd.read_csv(
        region_file,
        names=['chrom', 'second', 'region', 'start', 'end', 'sixth', 'strand', 'eighth', 'id_name_biotype'],
        sep='\t',
        header=None,
        dtype={
            'chrom': str, 'second': str, 'region': str, 'start': int, 'end': int, 'sixth': str, 'strand': str,
            'eight': str, 'id_name_biotype': str
        }
    )


def filter_cds_utr_ncrna(df_in):
    """Filter regions CDS, UTR5, UTR3 and ncRNA by size and trim."""
    utr5 = df_in.region == 'UTR5'
    cds = df_in.region == 'CDS'
    utr3 = df_in.region == 'UTR3'
    ncrna = df_in.region == 'ncRNA'
    longer = df_in.end - df_in.start >= 300
    short = df_in.end - df_in.start >= 100
    df_out = df_in[(utr5 & longer) | (cds & short) | (utr3 & longer) | ncrna].copy()
    df_out.loc[df_out['region'] == 'UTR3', ['start']] = df_out.start + 30
    df_out.loc[df_out['region'] == 'UTR5', ['end']] = df_out.end - 30
    df_out.loc[df_out['region'] == 'CDS', ['start']] = df_out.start + 30
    df_out.loc[df_out['region'] == 'CDS', ['end']] = df_out.end - 30
    return df_out


def filter_intron(df_in, min_size):
    """Filter intron regions to remove those smaller than min_size."""
    # remove regions shorter then min_size
    df_out = df_in[df_in.end - df_in.start >= min_size].copy()
    return df_out


def get_regions_map(regions_file):
    """Prepare temporary files based on GTF file that defines regions."""
    df_regions = pd.read_csv(
        regions_file, sep='\t', header=None,
        names=['chrom', 'second', 'region', 'start', 'end', 'sixth', 'strand', 'eighth', 'id_name_biotype'],
        dtype={
            'chrom': str, 'second': str, 'region': str, 'start': int, 'end': int, 'sixth': str, 'strand': str,
            'eight': str, 'id_name_biotype': str})
    df_intergenic = df_regions.loc[df_regions['region'] == 'intergenic']
    df_cds_utr_ncrna = df_regions.loc[df_regions['region'].isin(['CDS', 'UTR3', 'UTR5', 'ncRNA'])]
    df_intron = df_regions.loc[df_regions['region'] == 'intron']
    df_cds_utr_ncrna = filter_cds_utr_ncrna(df_cds_utr_ncrna)
    df_intron = filter_intron(df_intron, 100)
    to_csv_kwrgs = {'sep': '\t', 'header': None, 'index': None}
    df_intron.to_csv('{}intron_regions.bed'.format(TEMP_PATH), **to_csv_kwrgs)
    df_intergenic.to_csv('{}intergenic_regions.bed'.format(TEMP_PATH), **to_csv_kwrgs)
    df_cds_utr_ncrna.to_csv('{}cds_utr_ncrna_regions.bed'.format(TEMP_PATH), **to_csv_kwrgs)

def get_sequences(sites, fasta, fai, window_l, window_r, merge_overlaps=False):
    """Get genome sequences around positions defined in sites."""
    sites = pbt.BedTool(sites).sort()
    sites_extended = sites.slop(l=window_l, r=window_r, g=fai)  # noqa
    if merge_overlaps:
        sites_extended = sites_extended.merge(s=True)
    seq_tab = sites_extended.sequence(s=True, fi=fasta, tab=True)
    return [line.split("\t")[1].strip() for line in open(seq_tab.seqfn)]


def pos_count_kmer(seqs, k_length, window, kmer_list):
    """Gets number of occurences of each kmer for each position in a list of sequnces.
    
    Alternativly, if kmer_list is defined, it returns positional counts only for kmers in the list"""
    
    zero_counts = {pos: 0 for pos in range(-window, window + 1)}
    if kmer_list:
        possible_kmers = kmer_list
    else:
        possible_kmers = []
        for i in product('ACGT', repeat=k_length):
            possible_kmers.append("".join(i))
    kmer_pos_count = {x: zero_counts.copy() for x in possible_kmers}
    for sequence in seqs:
        for i in range(k_length, len(sequence) - k_length):
            kmer = sequence[i: i + k_length]
            relative_pos = i - window - k_length
            try:
                kmer_pos_count[kmer][relative_pos] += 1
            except KeyError:
                pass
    return kmer_pos_count


def get_positions(s, kmer):   
    def find_all(a_str, sub):
        """find indices of the substring in a string"""
        start = 0
        while True:
            start = a_str.find(sub, start)
            if start == -1: return
            yield start
            start += 1
    # for each sliding window we find which positions fit into each motif
    #sequence_positions = {}
    indices_extended = []
    indices = list(find_all(s, kmer))           
    for i in indices:
        indices_extended.extend(range(i, (i + len(kmer))))
    position = [1 if x in set(indices_extended) else 0 for x in range(len(s))]
    # number of positions is summed up into score for the current window
    # score of the window with max score is returned, called h_max 
    return position


def remove_chr(df_in, chr_sizes, chr_name='chrM'):
    """Remove chromosomes that are not in genome annotations.

    Also removes ``chr_name`` from DataFrame.
    """
    df_chr_sizes = pd.read_csv(
        chr_sizes, names=['chrom', 'end'], sep='\t', header=None, dtype={'chrom': str, 'end': int}
    )
    df_in = df_in[df_in['chrom'].isin(df_chr_sizes['chrom'].values)]
    return df_in[~(df_in['chrom'] == chr_name)]


def intersect(interval_file, s_file):
    """Intersect two BED files and return resulting BED file."""
    if interval_file:
        result = pbt.BedTool(s_file).intersect(
            pbt.BedTool(interval_file), s=True,
            nonamecheck=True,
        ).saveas()
    else:
        result = pbt.BedTool(s_file)
    if len(result) >= 1:
        return result


def intersect_merge_info(region, s_file):
    """Intersect while keeping information from region file."""
    interval_file = REGIONS_MAP[region]
    try:
        df_1 = intersect(interval_file, s_file).to_dataframe(
            names = ['chrom', 'start', 'end', 'name', 'score', 'strand'],
            dtype={'chrom': str, 'start': int, 'end': int, 'name': str, 'score': float, 'strand': str}
        )
        df_1 = df_1.groupby(['chrom', 'start', 'end', 'strand'], as_index=False)['score'].sum(axis=0)
        df_1['name'] = '.'
        df_2 = intersect(s_file, interval_file).to_dataframe(
            names = ['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attributes'],
            dtype={'seqname': str, 'source': str, 'feature': str, 'start': int, 'end': int, 'score': str,
                'strand': str, 'frame': str, 'attributes': str}
        )
        df_2.drop_duplicates(subset=['seqname', 'start', 'end', 'strand'], keep='first')
    except AttributeError:
        print(f'{s_file} might have no sites in {region}')
        return
    df_2 = df_2.drop(columns=['source', 'score', 'frame', 'start']).rename(index=str, columns={"seqname": "chrom"})
    return pd.merge(df_1, df_2, on=['chrom', 'strand', 'end'])

def cut_per_chrom(chrom, df_p, df_m, df_peaks_p, df_peaks_m):
    """Split data by strand then apply pandas cut to each strand.

    Pandas cut uses IntervalIndex (done from the peaks file) to
    assign each site its peak. Finally merges strands.
    """
    df_temp_p = df_peaks_p[df_peaks_p['chrom'] == chrom].copy()
    df_temp_m = df_peaks_m[df_peaks_m['chrom'] == chrom].copy()
    df_xl_p = df_p[df_p['chrom'] == chrom].copy()
    df_xl_m = df_m[df_m['chrom'] == chrom].copy()
    left_p = np.array(df_temp_p['start'])
    right_p = np.array(df_temp_p['end'])
    left_m = np.array(df_temp_m['start'])
    right_m = np.array(df_temp_m['end'])
    interval_index_p = pd.IntervalIndex.from_arrays(left_p, right_p, closed='left')
    interval_index_m = pd.IntervalIndex.from_arrays(left_m, right_m, closed='left')
    df_xl_p['cut'] = pd.cut(df_xl_p['start'], interval_index_p)
    df_xl_m['cut'] = pd.cut(df_xl_m['start'], interval_index_m)
    return pd.concat([df_xl_p, df_xl_m], ignore_index=True)


def cut_sites_with_region(df_sites, df_region):
    """Find peak interval the crosslinks belong to."""  
    df_p = df_sites[df_sites['strand'] == '+'].copy()
    df_m = df_sites[df_sites['strand'] == '-'].copy()
    df_region_p = df_region[df_region['strand'] == '+'].copy()
    df_region_m = df_region[df_region['strand'] == '-'].copy()
    df_cut = pd.DataFrame(columns=['chrom', 'start', 'end', 'name', 'score', 'strand', 'feature', 'attributes', 'cut'])
    for chrom in set(df_region['chrom'].values):
        df_temp = cut_per_chrom(chrom, df_p, df_m, df_region_p, df_region_m)
        df_temp = df_temp[df_cut.columns]
        df_cut = pd.concat([df_cut, df_temp], ignore_index=True)
    return df_cut.dropna(axis=0)


def percentile_filter_xlinks(df_in, percentile=0.7):
    """Calculate threshold and filter sites by it."""
    df_in['cut'] = df_in['cut'].astype(str)
    df_in['quantile'] = df_in['cut'].map(df_in.groupby('cut').quantile(q=percentile)['score'])
    df_in = df_in[df_in['score'] > df_in['quantile']]
    return df_in[['chrom', 'start', 'end', 'name', 'score', 'strand', 'feature', 'attributes']]


def get_threshold_sites(s_file, percentile=0.7):
    """Apply crosslink filtering based on dynamical thresholds.

    Regions for thresholds are defined as follows: introns and
    intergenic regions are each idts own region, for CDS, UTR and ncRNA
    each gene is a region. After region determination threshold based on
    percentile are applied and finally threshold crosslinks sites are
    sorted.
    """
    df_out = pd.DataFrame(columns=['chrom', 'start', 'end', 'name', 'score', 'strand', 'feature', 'attributes'])
    for region in REGIONS_QUANTILE:
        print(f'Thresholding {region}')
        df_reg = intersect_merge_info(region, s_file)
        print(f'lenght of df_reg for {region} is: {len(df_reg)}')
        if df_reg is None:
            continue
            
        if region == 'cds_utr_ncrna':
            df_reg.name = df_reg.attributes.map(lambda x: x.split(';')[1].split(' ')[1].strip('"'))
            df_reg['quantile'] = df_reg['name'].map(df_reg.groupby(['name']).quantile(q=percentile)['score'])
            df_filtered = df_reg[df_reg['score'] > df_reg['quantile']].drop(columns=['quantile'])
            df_out = pd.concat([df_out, df_filtered], ignore_index=True, sort=False)
        if region in ['intron', 'intergenic']:
            df_region = parse_region_to_df(REGIONS_MAP[region])
            df_cut = cut_sites_with_region(df_reg, df_region)
            df_filtered = percentile_filter_xlinks(df_cut)
            df_out = pd.concat([df_out, df_filtered], ignore_index=True, sort=False)
    return df_out.sort_values(by=['chrom', 'start', 'strand'], ascending=[True, True, True]).reset_index(drop=True)




def get_all_sites(s_file):
    """Get crosslink data into appropriate dataframe without thresholding."""
    df_out = pd.DataFrame(columns=['chrom', 'start', 'end', 'name', 'score', 'strand', 'feature', 'attributes'])
    for region in REGIONS_QUANTILE:
        df_reg = intersect_merge_info(region, s_file)
        if df_reg.empty:
            continue
        if region == 'cds_utr_ncrna':
            df_reg.name = df_reg.attributes.map(lambda x: x.split(';')[1].split(' ')[1].strip('"'))
            df_reg['quantile'] = None
            df_out = pd.concat([df_out, df_reg], ignore_index=True, sort=False)
        if region in ['intron', 'intergenic']:
            df_region = parse_region_to_df(REGIONS_MAP[region])
            df_cut = cut_sites_with_region(df_reg, df_region)
            df_filtered = df_cut[['chrom', 'start', 'end', 'name', 'score', 'strand', 'feature', 'attributes']]
            df_out = pd.concat([df_out, df_filtered], ignore_index=True, sort=False)
    return df_out.sort_values(by=['chrom', 'start', 'strand'], ascending=[True, True, True]).reset_index(drop=True)



def run(sites_files_paths_list, peak_file_path_list, kmer_list_input, region_list_input, kmer_length, 
        genome, genome_fai, regions_file, smoot, percentile=None):
    """Run the script"""
    global TEMP_PATH
    TEMP_PATH = './TEMP{}/'.format(randint(10 ** 6, 10 ** 7))
    make_temp = local["mkdir"]
    make_temp(TEMP_PATH)
    os.makedirs('./results/', exist_ok=True)
    
    kmer_list_input = [kmer.replace('U', 'T') for kmer in kmer_list_input]

    df_out = pd.DataFrame()
    
    for sites_file, peak_file in zip(sites_files_paths_list, peak_file_path_list):
        print(f'Analysing: {sites_file},    {peak_file}')
        sample_name = get_name(sites_file)
        get_regions_map(regions_file)
        global REGIONS_MAP
        REGIONS_MAP = {
            'intron': '{}intron_regions.bed'.format(TEMP_PATH),
            'intergenic': '{}intergenic_regions.bed'.format(TEMP_PATH),
            'cds_utr_ncrna': '{}cds_utr_ncrna_regions.bed'.format(TEMP_PATH)
        }
        if percentile:
            print('Getting thresholded crosslinks')
            df_txn = get_threshold_sites(sites_file, percentile=percentile)                
            print(f'{len(df_txn)} thresholded crosslinks')
        else:
            print('Getting all crosslinks')
            df_txn = get_all_sites(sites_file)

        for region in region_list_input:
            # Parse sites file and keep only parts that intersect with given region
            df_sites = df_txn.loc[df_txn['feature'].isin(REGION_SITES[region])]
            if percentile:
                print(f'{len(df_sites)} thresholded sites on {region}')
            else:
                print(f'{len(df_sites)} total sites on {region}')

            sites = pbt.BedTool.from_dataframe(
            df_sites[['chrom', 'start', 'end', 'name', 'score', 'strand']])
            # scan surounding of sites to obtain positional distributions for each kmer in input list
            sequences = get_sequences(sites, genome, genome_fai, 150 + kmer_length, 150 + kmer_length)
            seq_pos_all_kmer = {}
            for kmer in kmer_list_input:
                seq_pos = [get_positions(s, kmer) for s in sequences]
                seq_pos_all_kmer[kmer] = seq_pos
            test_df = pd.DataFrame.from_dict(seq_pos_all_kmer)
            df_concat = pd.DataFrame()
            for kmer in kmer_list_input:
                df_temp = test_df[kmer].apply(pd.Series)
                df_concat = pd.concat([df_concat, df_temp])
            
            df_concat.rename(columns=(lambda x: x - 150 - kmer_length), inplace=True)
            # define name of column
            mean_col = '{}_{}'.format(sample_name, region)
            # calculate average of kmers of interest for each position and save results in new column
           # df_kpc[mean_col] = df_kpc.mean(axis=1)
            # copy the column of average counts to dataframe used for plotting
            df_out[mean_col] = df_concat.mean()
            # smoothen
            df_smooth = df_out.rolling(smoot, center=True, win_type='triang').mean()
            # slicing drops edge values that get NaN due to rolling mean
            df_smooth = df_smooth.iloc[int(smoot / 2): -(int(smoot / 2) + 1), :]
            df_smooth = df_smooth * 100
    # different cases of plotting the data, depending on how many plots need to be generated    
    plot_list = []
    # if there is just one file we plot distributions for each region on the same plot
    if len(sites_files_paths_list) == 1:
        df_plot = df_smooth
        plot_list.append(df_plot)
    # if there are multiple file but single region we plot all files on same plot
    elif (len(sites_files_paths_list) > 1) and (len(region_list_input) == 1):
        df_plot = df_smooth
        plot_list.append(df_plot)
    # if there is more then one file and region we plot different files together for each region
    elif (len(sites_files_paths_list) > 1) and (len(region_list_input) > 1):
        plots_list = []
        for r in region_list_input:
            columns = [x for  x in df_smooth.columns if r in x]
            df_plot = df_smooth[columns]
            plot_list.append(df_plot)
    
    # save a tsv file used for plotting, which contains average counts of kmers of interest for each position
    kmer_list_name = [kmer.replace('T', 'U') for kmer in kmer_list_input]
    outfile_name = "_".join(kmer_list_name[:4])
    df_out.to_csv(f'./results/{sample_name}_{outfile_name}.tsv', sep='\t', float_format='%.8f')
    
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    lineplot_kwrgs = {'palette': "tab10", 'linewidth': 1, 'dashes': False, }
    for p in plot_list:
        plt.figure()
        sns_plot = sns.lineplot(data = p, **lineplot_kwrgs)
        plt.ylabel('Coverage (%)')
        plt.xlabel('Positions of kmer start relative to crosslinks')
        plt.title('Coverage of {} motif group in {}'.format(outfile_name, region))
        if (len(region_list_input) == 1):
            sns_plot.figure.savefig('./results/positional_distribution.png')
        else:
            region = p.columns[0].split('_')[-1]
            sns_plot.figure.savefig('./results/{}_positional_distribution.png'.format(region))
        plt.show()
        plt.close()

    remove_tmp = local["rm"]   
    remove_tmp("-rf", TEMP_PATH)