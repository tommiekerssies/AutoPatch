#%%
num_stages = 4

stem_widths = [64]

body_width_min = [64, 64, 128, 256]
body_width_max = [64, 128, 256, 512]
body_width_step = [0, 64, 128, 256]
body_widths = [
    range(
        body_width_min[i],
        body_width_max[i] + body_width_step[i],
        body_width_step[i],
    ) if body_width_step[i] > 0 else [body_width_min[i]]
    for i in range(num_stages)
]

body_depth_min = [1, 1, 1, 1]
body_depth_max = [2, 2, 2, 2]
body_depth_step = [1, 1, 1, 1]
body_depths = [
    range(
        body_depth_min[i],
        body_depth_max[i] + body_depth_step[i],
        body_depth_step[i],
    )
    for i in range(num_stages)
]

from itertools import product
unpruned_archs = []
unpruned_archs.extend(
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
        overhead_as_strings={}
    )
    for params in product(stem_widths, *body_widths, *body_depths)
)
print(f"unpruned_archs: {len(unpruned_archs)}")

# now keep only archs with monotonically increasing widths
archs = []
for arch in unpruned_archs:
    last_checked_width = arch["arch"]["encoder_q"]["stem"]["width"]
    width_monotonically_increasing = True
    for width in arch["arch"]["encoder_q"]["body"]["width"]:
        if width < last_checked_width:
            width_monotonically_increasing = False
            break
        last_checked_width = width
    if width_monotonically_increasing:
        archs.append(arch)

print(f"archs: {len(archs)}")

#%%
from app.lightning_module.supernet import SuperNet
lm = SuperNet(
    body_width=body_width_max, 
    body_depth=body_depth_max, 
    stem_width=64,
    resume_run_id=None,
    weights_file=None,
).cuda()

from lib.gaia.flops_counter import flops_to_string, get_model_complexity_info, params_to_string
for arch in archs:
    lm.model.manipulate_arch(arch["arch"])
    arch["overhead"]["flops"], arch["overhead"]["params"] = \
        get_model_complexity_info(
            lm.model.backbone,
            (3, 224, 224), # TODO: make input shape a parameter
            print_per_layer_stat=False,
            as_strings=False,
        )  
    arch["overhead_as_strings"]["flops"], arch["overhead_as_strings"]["params"] = \
        flops_to_string(arch["overhead"]["flops"]), params_to_string(arch["overhead"]["params"])

#%%
from pandas import json_normalize
file = open('model_space.csv', 'w')
json_normalize(archs).to_csv(file)