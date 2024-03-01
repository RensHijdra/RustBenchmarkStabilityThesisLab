import csv
import json
import os
import re
import typing
from collections import defaultdict
from pathlib import Path

import statsmodels.api as sm
import numpy
import numpy as np
import scipy
import scipy as sp
import seaborn
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

CONSUMPTION_PATH = "raw/data"
COVERAGE_PATH = "../scrape-crates/coverage/"


def custom_mquantiles_cimh_hd(data, prob=[0.25, 0.50, 0.75], alpha=0.05, axis=None):
    alpha = min(alpha, 1 - alpha)
    z = scipy.stats.norm.ppf(1 - alpha / 2.)
    xq = scipy.stats.mstats.hdquantiles(data, prob, axis=axis)
    smj = scipy.stats.mstats.mjci(data, prob, axis=axis)
    return xq - z * smj, xq + z * smj


def rciw_boot(data: numpy.ndarray, cl: float) -> float:
    bs = scipy.stats.bootstrap((data,), statistic=scipy.stats.mstats.hdmedian, n_resamples=10000, confidence_level=cl)
    # print()
    return (bs.confidence_interval.high - bs.confidence_interval.low) / np.mean(data)


def rciw_mjhd(data: numpy.ndarray, cl: float) -> float:
    ci = custom_mquantiles_cimh_hd(data, prob=[0.5], alpha=cl)
    rciw = (ci[1] - ci[0]) / scipy.stats.mstats.hdmedian(data)
    return rciw[0]


def rmad_hd(data: np.array) -> float:
    median = numpy.ma.getdata(scipy.stats.mstats.hdmedian(data[~np.isnan(data)], axis=None))
    # print(median)
    mad = scipy.stats.mstats.hdmedian(np.array([np.abs(point - median) for point in data[~np.isnan(data)]]))
    # abs_devs = np.array([scipy.stats.median_abs_deviation(data, center=center, nan_policy='omit')])
    # mad = scipy.stats.mstats.hdmedian(abs_devs)
    rmad = mad / median
    return rmad 


def collect_coverage_paths() -> typing.Iterator[Path]:
    for dir_path, dir_names, filenames in os.walk(COVERAGE_PATH):
        if not dir_names:  # If we are in a leaf directory
            try:
                # Get the most recent csv from the directory
                yield dir_path + '/' + sorted(filter(lambda file: '.csv' in file, filenames))[-1]
            except IndexError:
                continue  # No CSV file present


def get_project_and_benchmark_from_coverage_path(coverage: Path) -> (str, str):
    pattern = re.compile(r".*?/coverage/([\w\-:._…\s&]+)/([\w+\-/_…:\s&]+)/")
    group = pattern.match(str(coverage)).groups()
    return group


def get_project_and_benchmark_from_consumption_path(consumption: Path) -> (str, str):
    pattern = re.compile(r".*?[/\\][0-9]+[/\\]([\w-]+)[/\\]criterion[/\\](.+?)[/\\]new[/\\]sample.json")
    group = pattern.match(str(consumption)).groups()
    return str(group[0]), str(group[1])


def get_energy_files_for_benchmark(project: str, benchmark_id: str) -> typing.List[Path]:
    return list(Path(CONSUMPTION_PATH).rglob(f"{project}/criterion/{benchmark_id}/new/sample.json"))


def read_features():
    return {row.strip(): 0 for row in Path("./features.csv").open().readlines() if not row.startswith('#')}


def combine_coverage_data():
    # Read the features file while skipping comments
    empty_data = read_features()
    # Create a CSV writer with features as headers, for once and for
    print(list(empty_data.keys()))
    once_count_writer = csv.DictWriter(Path('./datapoints_once.csv').open("w"), fieldnames=list(empty_data.keys()),
                                       delimiter=';')
    execution_count_writer = csv.DictWriter(Path('./datapoints_count.csv').open("w"),
                                            fieldnames=list(empty_data.keys()), delimiter=';')
    once_count_writer.writeheader()
    execution_count_writer.writeheader()

    for cov in collect_coverage_paths():
        (proj, bench) = get_project_and_benchmark_from_coverage_path(Path(cov))

        execution = empty_data.copy()
        execution['_id'] = proj + '/' + bench
        count = empty_data.copy()
        count['_id'] = proj + '/' + bench

        for [key, value] in csv.reader(Path(cov).open(), delimiter=','):
            # check if k is contained in a key of default dict
            for feature in empty_data.keys():
                # print("feat", feature)
                if feature in key:
                    value = int(value)
                    if key.startswith("once"):
                        execution[feature] += value
                    elif key.startswith("count"):
                        count[feature] += value
                    else:
                        print("Adding feature")
                        execution[feature] += value
                        count[feature] += value
                    break

        once_count_writer.writerow(count)
        execution_count_writer.writerow(execution)


#
# def parse_data():
#     energy_samples = list(Path(CONSUMPTION_PATH).rglob("new/sample.json"))
#     pattern = re.compile(r".*?/(\w+)/criterion/(.+?)/new/sample.json")
#
#     dataset = defaultdict(lambda: [0] * len(energy_samples))
#
#     for bidx, benchmark in enumerate(energy_samples):
#         # print(benchmark)
#         with open(benchmark) as sams:
#
#             project, bench_id, *_ = pattern.match(str(benchmark)).groups()
#
#             if not coverage_path.exists():
#                 # matrix['_id'][bidx] = bidx
#                 # matrix['__rmad'][bidx] = 0.0
#                 continue
#
#             data = json.load(sams)
#             iters = data['iters']
#             power = data['times']  # criterion calls it times, but it is actually measured power consumption
#             energy_per_iter = list(map(lambda p: p[0] / p[1], zip(power, iters)))
#             rmad = rmad_hd(energy_per_iter)
#
#             latest_coverage = list(coverage_path.glob("*.csv"))[-1]
#
#             data = dict(
#                 map(lambda line: (line.split(',')[0], float(line.split(',')[1])), open(latest_coverage).readlines()))
#             data['__rmad'] = rmad * 100
#
#             for key in data.keys():
#                 if key == '' or ('::' in key and not any(key.startswith(mod) for mod in ["std", "core", "alloc"])):
#                     continue
#                 dataset[key][bidx] = data[key]
#
#     return dataset


def get_normalized_mj_from_file(path: Path):
    data = json.load(path.open())
    if len(data['times']) != 300:
        return np.array([])
    # mj = list(map(lambda joules: joules * 1000, filter(lambda joules: joules < 2**30, data['times'])))
    # [xv if c else yv for c, xv, yv in zip(condition, x, y)]
    # fix a mistake in collecting measurements where a u32 contained in a u64 was used with wrapping subtraction
    false_energies = np.array(data['times'])
    false_energies[false_energies > 2] = np.NaN
    mean = np.nanmean(false_energies)
    false_energies[np.where(np.isnan(false_energies))] = mean
    # if (false_energies > 2).any():
    #     print(str(path) + " contains too high energy")
    end_start_difference = false_energies / (0.5**15)
    
    ints = np.array(list(map(lambda diff: int(diff) & 0xFFFF_FFFF, end_start_difference)))
    
    fixed = ints * (0.5**15)

    return numpy.array(np.array(fixed) * 1000) / np.array(data['iters']) 

def collect_energy_data_paths():
    return Path(CONSUMPTION_PATH).rglob("new/sample.json")


def collect_energy_to_csv(outfile: str, confidence: float = 0.99):
    benchmarks = defaultdict(list)
    for path in collect_energy_data_paths():
        (proj, bench) = get_project_and_benchmark_from_consumption_path(path)
        if proj == 'tracing':
            continue
        benchmarks[proj + '/' + bench].append(get_normalized_mj_from_file(path).tolist())
    with open(outfile, 'w') as output:
        print("bench", "rmad", "rciw_mjhd", "rciw_boot", sep=";", file=output, end='\n', flush=True)

        for k, data in benchmarks.items():
            arr = numpy.array(data)
            rmad = rmad_hd(arr)
            # mjhd = rciw_mjhd(arr, confidence)
            # boot = rciw_boot(arr, confidence)
            # print(";".join([k, str(rmad), str(mjhd), str(boot)]) , file=output, end='\n', flush=True)
            print(";".join([k, str(rmad)]), file=output, end='\n', flush=True)


def plt_energies(file: str):
    df = pd.read_csv(file, delimiter=';')
    # reader = csv.reader(file, delimiter=';')
    # next(reader)
    # array = numpy.array([[float(item) for item in row[1:]] for row in reader])
    df.drop(columns=['bench', 'rmad'])
    df *= 100
    stripplot = sns.stripplot(data=df, jitter=0.35, alpha=0.1)
    sns.despine(bottom=True)
    plt.yscale('log')
    plt.xlabel("Variability measures")
    plt.ylabel("Relative variability (%)")
    # plt.savefig('energy.png', format='png', dpi=500)
    #
    # ax = sns.violinplot(data=df, )
    # sns.despine(bottom=True)
    # ax.tick_params(bottom=False)
    # plt.savefig("energy95.png", dpi=500)
    # plt.show()


def energy_regression_plot():
    df = pd.read_csv('third_energy.csv', delimiter=';')
    new_col = dict()
    inter_measurement = []
    for path in collect_energy_data_paths():
        (proj, bench) = get_project_and_benchmark_from_consumption_path(path)
        d = dict()
        data = json.load(path.open())
        d['period'] = 10.0 / float(data['iters'][0])
        d['rmad'] = rmad_hd(numpy.array(data['times']))
        d['rciw_mjhd'] = rciw_mjhd(numpy.array(data['times']), 0.99)
        # d['rciw_boot'] = rciw_boot(numpy.array(data['times']), 0.99)
        new_col[proj + '/' + bench] = 10.0 / float(data['iters'][0])
        inter_measurement.append(d)
        # return numpy.array(data['times']) * 1000 / /numpy.array(data['iters'])
    df['period'] = df['bench'].map(new_col)
    df.drop(columns='bench')
    sns.scatterplot(data=df, legend=True)
    plt.ylabel = "variability (%)"
    plt.xlabel = "avg benchmark duration (ms)"
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

    frame = pd.DataFrame(inter_measurement)
    sns.scatterplot(x=frame['period'], y=frame['rmad'])
    sns.scatterplot(x=frame['period'], y=frame['rciw_mjhd'])
    plt.xscale('log')
    plt.yscale('log')
    plt.show()


if __name__ == '__main__':
    # for consumption in collect_energy_data_paths():
    #
    #     (proj, bench) = get_project_and_benchmark_from_consumption_path(consumption)
    #     # print(proj, bench)
    #     energy_files = get_energy_files_for_benchmark(proj, bench)
    #
    #     if not energy_files:
    #         continue  # if unable to find energy files, continue
    #     energy_entries =pd.DataFrame({file: get_normalized_mj_from_file(Path(file)) for file in energy_files})
    # print(energy_entries.T)
    # print(len(energy_entries))
    # print(list(len(e) for e in energy_entries))
    # rmad = rmad_hd(energy_entries)
    # energy_entries.sort()
    # energy_entries = energy_entries.T
    # print(proj, bench)
    # energy_entries = np.log10(energy_entries)
    # plt.boxplot(energy_entries)
    # plt.ylim([10**-6, 10**-3])

    # energy_entries = np.log(energy_entries)
    # sm.qqplot(energy_entries, line='45', fit=True)
    # plt.scatter(energy_entries)
    # sns.kdeplot(energy_entries.T, bw_method=0.4, alpha=0.5, legend=False)
    # plt.show()
    # flatten = energy_entries.to_numpy().flatten()
    # if sp.stats.normaltest(flatten).pvalue > 0.0005 :
    #     print("normal", proj, bench)
    # if sp.stats.normaltest(np.log(flatten)).pvalue > 0.0005:
    #     print("lognormal", proj, bench)
    #     print("rmad", rmad)
    #     print("mjhd", rciw_mjhd(energy_entries, 0.95))
    #     print("boot", rciw_boot(energy_entries, 0.95))
    #     print("")
    # collect_energy_to_csv("rmads.csv", 0.95)
    combine_coverage_data()

    # plt_energies('third_energy.csv')
    # print(energy_regression_plot())
