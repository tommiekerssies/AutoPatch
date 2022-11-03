from lib.gaia.flops_counter import get_model_complexity_info

# TODO: force widths to be doubling
# TODO: dont save models bigger in flops than rn18
# TODO: set max width to standard rn width, so we can load imagenet weights for supernet

num_stages = len(self.hparams.body_width_step)

body_widths = [
    range(
        self.hparams.body_width_min[i],
        self.hparams.body_width[i] + self.hparams.body_width_step[i],
        self.hparams.body_width_step[i],
    )
    for i in range(num_stages)
]

body_depths = [
    range(
        self.hparams.body_depth_min[i],
        self.hparams.body_depth[i] + self.hparams.body_depth_step[i],
        self.hparams.body_depth_step[i],
    )
    for i in range(num_stages)
]

stem_widths = [
    range(
        self.hparams.stem_width_min[i],
        self.hparams.stem_width[i] + self.hparams.stem_width_step[i],
        self.hparams.stem_width_step[i],
    )
    for i in range(num_stages)
]

architectures: List[Any] = [
    dict(
        arch=dict(
            encoder_q=dict(
                stem=dict(width=params[0]),
                body=dict(
                    width=params[1 : num_stages + 1],
                    depth=params[num_stages + 1 :],
                ),
            ),
        ),
        overhead={},
    )
    for params in product(stem_widths, *body_widths, *body_depths)
]

self.cuda()
init_process_group(backend="nccl")
for arch in architectures:
    self.model.manipulate_arch(arch["arch"])
    (  # TODO: make input shape a parameter
        arch["overhead"]["flops"],
        arch["overhead"]["params"],
    ) = get_model_complexity_info(
        self.model.backbone,
        (3, 224, 224),
        print_per_layer_stat=False,
        as_strings=False,
    )
    print(arch)
