"""Generic heterogeneous 6x6 periodic qMERA scale-plan example."""

from pepsy.optimizers.qmera import (
    QMeraBuilder,
    QMeraDisentanglerSpec,
    QMeraIsometrySpec,
    QMeraScaleSpec,
)


def main():
    scales = (
        QMeraScaleSpec(
            isometry=QMeraIsometrySpec(block_shape=(2, 2)),
            disentangler=QMeraDisentanglerSpec(
                block_shape=(2, 2),
                placement="boundary-square",
            ),
        ),
        QMeraScaleSpec(
            isometry=QMeraIsometrySpec(block_shape=(3, 3)),
            disentangler=QMeraDisentanglerSpec(
                block_shape=3,
                orientation="vertical",
                placement="within-block",
                circuit_depth=3,
            ),
        ),
    )
    schedule = QMeraBuilder(
        shape=(6, 6),
        boundary="periodic",
        scales=scales,
    ).build_schedule()

    print("active sites:", [len(layer.input_sites) for layer in schedule.layers], "->", len(schedule.top_sites))
    print("isometry blocks:", [len(layer.isometry_blocks) for layer in schedule.layers])
    print("disentangler blocks:", [len(layer.disentangler_blocks) for layer in schedule.layers])


if __name__ == "__main__":
    main()
