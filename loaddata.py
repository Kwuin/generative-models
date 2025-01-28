from datasets import load_dataset



#
cache_dir = "/scratch/user/jkunlin/generative-models/huggingface_cache"

#
dataset = load_dataset("kakaobrain/coyo-700m", cache_dir=cache_dir)

# 在 loaddata.py 中添加过滤逻辑
dataset = dataset.filter(
    lambda x: x["aesthetic_score_laion_v2"] > 5.0,  # 保留美学评分 >5 的样本
    num_proc=16  # 并行处理
)

print("dataset structure", dataset)
print("\nfirst sample", dataset["train"][0])

