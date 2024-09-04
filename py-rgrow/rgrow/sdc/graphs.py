from typing import Iterable, Sequence
from .anneal import Anneal, AnnealOutputs
from .sdc import SDC
from .reporter_methods import ReportingMethod

import matplotlib.pyplot as plt


MIN = 60
HOUR = MIN * 60


def graph_system(
    system: SDC,
    anneal_output: AnnealOutputs,
    method: ReportingMethod,
    # TODO: Fix this for windows
    path: str = "/tmp/sdc_image.png",
):
    measurement = method.reporter_method(anneal_output)

    times, temps = anneal_output.anneal.gen_arrays()
    times_hours = times / HOUR

    plt.clf()

    plt.plot(times_hours, measurement, label=system.name)
    plt.xlabel("Time (hours)")
    plt.ylabel(method.desc)
    plt.ylim(0.0, 1.1)

    plt.legend()

    # Now plot the temperature
    plt2 = plt.twinx()
    plt2.plot(times_hours, temps, "k--", label="temperature C")
    plt2.set_ylabel("temperature")

    plt.savefig(path)


def graph_system_with_many_reporting_methods(
    system: SDC,
    anneal_output: AnnealOutputs,
    methods: list[ReportingMethod],
    # TODO: Fix this for windows
    path: str = "/tmp/sdc_image.png",
):
    plt.clf()
    times, temps = anneal_output.anneal.gen_arrays()
    times_hours = times / HOUR

    for method in methods:
        measurement = method.reporter_method(anneal_output)
        plt.plot(times_hours, measurement, label=method.desc)

    plt.xlabel("Time (hours)")
    plt.ylabel("Method Depended Error")
    plt.ylim(0.0, 1.1)

    plt.legend()

    # Now plot the temperature
    plt2 = plt.twinx()
    plt2.plot(times_hours, temps, "k--", label="temperature C")
    plt2.set_ylabel("temperature")

    plt.title(system.name)

    plt.savefig(path)


def run_and_graph_system(
    system: SDC,
    anneal: Anneal,
    method: ReportingMethod,
    # TODO: Fix this for windows
    path: str = "/tmp/sdc_image.png",
):
    graph_system(system, system.run_anneal(anneal), method, path)


def run_and_graph_system_with_many_reporting_methods(
    system: SDC,
    anneal: Anneal,
    methods: list[ReportingMethod],
    # TODO: Fix this for windows
    path: str = "/tmp/sdc_image.png",
):
    graph_system_with_many_reporting_methods(
        system, system.run_anneal(anneal), methods, path
    )


def graph_many_systems_with(
    outputs: Sequence[AnnealOutputs],
    method: ReportingMethod,
    # TODO: Fix this for windows
    path: str = "/tmp/sdc_image.png",
    title: str | None = None,
):
    plt.clf()
    times, temps = outputs[0].anneal.gen_arrays()
    times_hours = times / HOUR

    for o in outputs:
        measurement = method.reporter_method(o)
        plt.plot(times_hours, measurement, label=o.system.name)

    plt.xlabel("Time (hours)")
    plt.ylabel(method.desc)
    plt.ylim(0.0, 1.1)

    plt.legend()

    # Now plot the temperature
    plt2 = plt.twinx()
    plt2.plot(times_hours, temps, "k--", label="temperature C")
    plt2.set_ylabel("temperature")

    if title is not None:
        plt.title(title)

    plt.savefig(path)


def run_and_graph_many_systems_with(
    systems: Iterable[SDC],
    anneal: Anneal,
    method: ReportingMethod,
    # TODO: Fix this for windows
    path: str = "/tmp/sdc_image.png",
    title: str | None = None,
):
    outs = [s.run_anneal(anneal) for s in systems]
    graph_many_systems_with(outs, method, path, title)
