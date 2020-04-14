import h5py
import click


from argparse import ArgumentParser


def get_labels(h5root):
    labels = h5root["ObjectClassification/LabelNames"][()]
    return {k: v.decode("utf-8") for k, v in enumerate(labels, start=1)}


def print_verbose(results):
    for fname, labels, counts in results:
        print(f"filename: {fname}; total count: {sum(sum(count) for count in counts.values())}")
        for label in labels:
            print(f"\tlabel {labels[label]}: {sum(counts[label])}")


@click.command()
@click.argument("object_classification_files", type=click.Path(exists=True), nargs=-1)
def count_annotations(object_classification_files):
    results = []
    for fname in object_classification_files:
        f = h5py.File(fname, "r")
        labels = get_labels(f)
        counts = {k: list() for k in labels}
        lane_annotations = f["ObjectClassification/LabelInputs"]
        for lane in lane_annotations:
            timesteps = lane_annotations[lane]
            for timestep in timesteps:
                current = timesteps[timestep][()].astype("int")
                for k in labels.keys():
                    counts[k].append(sum(current == k))
        results.append((fname, labels, counts))

    print_verbose(results)


if __name__ == "__main__":
    count_annotations()
