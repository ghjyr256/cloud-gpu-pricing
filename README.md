# best cloud gpu: real hourly pricing, VRAM math, and how to pick the right GPU for training vs inference

The best cloud GPU is the cheapest card whose memory fits your model and whose billing model matches the way you actually work. That's the whole decision, and it's worth stating up front because the market has turned genuinely confusing: the same NVIDIA H100 rents for roughly $1.30/hr on a peer-to-peer marketplace, $2.99/hr on mainstream clouds like Vultr, and works out to about $12.30 per GPU on AWS once you divide a ~$98/hr p5.48xlarge instance by eight. Same silicon. Four to nine times the price depending on who you compare.

This guide sorts out which of those numbers applies to you. It covers verified on-demand pricing across the main providers, the VRAM math for 70B and 405B models, and a closer look at Vultr, which occupies a useful middle position: specialist-style hourly rates on a mainstream cloud that also handles your storage, VPC and Kubernetes. Along the way you'll get the billing details — per-second vs hourly, egress fees, monthly caps — that decide your real invoice and that most comparison posts skip entirely.

> **Quick picks**
>
> - **Best all-rounder on a mainstream cloud: Vultr.** Flat, published on-demand pricing — H100 80 GB at $2.99/hr, L40S 48 GB at $1.43/hr — with self-serve deployment and no marketplace lottery. 👉 [Check Vultr's current cloud GPU pricing](https://www.vultr.com/pricing/?ref=7268283)
> - **Cheapest raw compute: Vast.ai's marketplace,** H100 from around $1.30/hr. You trade away consistent hardware and any SLA.
> - **Multi-node training runs: Lambda Labs or CoreWeave,** where InfiniBand interconnects are part of the package.
> - **Hands-off inference: RunPod's serverless endpoints** or a managed API service like Together AI.

## Why the same H100 costs $1.30 or $12.30

The price gap comes from what's wrapped around the chip. Hyperscalers bundle enterprise procurement, compliance tooling, quota approval queues and bandwidth surcharges into the rate. Specialized GPU clouds stripped most of that out, and as supply caught up with demand, H100 prices fell 40–50% from their early-2025 levels. The expensive part of renting from AWS is the machinery around the GPU, and if you don't need that machinery, you're paying for it anyway.

Here's the current landscape, cross-checked across recent pricing comparisons:

| Provider | H100, on-demand | Billing granularity | What you're paying for |
| --- | --- | --- | --- |
| Vast.ai | from ~$1.30/hr | per minute | peer-hosted marketplace; lowest rates, uneven hardware, no SLA |
| RunPod | $1.89–2.79/hr by tier | per second | serverless endpoints, scale-to-zero, cold starts under 200 ms for many requests |
| Lambda Labs | ~$2.00–2.50/hr | per minute | InfiniBand-connected 1-Click Clusters, reserved pricing around 50% off |
| io.net | $2.10–3.50/hr | on-demand | decentralized pool of 320,000+ GPUs across 130+ countries |
| CoreWeave | ~$4.76–6.16/hr | committed contracts | bare-metal Kubernetes and non-blocking 3.2 Tbps InfiniBand for 70B–405B pretraining |
| AWS (p5.48xlarge) | ≈$12.30 per H100 | per second | the full enterprise ecosystem: IAM, VPC, procurement, quota queues |

Marketplace prices shift by the hour, naturally; the fixed-price providers quote what you'll actually see at checkout. Two honest caveats apply to that table. Vast.ai's floor comes with hardware quality that varies host to host, and CoreWeave mostly sells committed contracts rather than an on-demand slider.

Egress is the second trap. The legacy hyperscalers typically add $0.08–$0.12 per GB when you pull data out; some specialists, Hyperstack among them, charge nothing at all. If you're serving model weights, shipping training data, or pushing large inference batches around, that line item quietly dwarfs small differences in hourly rate.

## Vultr's cloud GPU lineup, plan by plan

Vultr rents NVIDIA GPUs by the hour on standard cloud instances — no bidding, no auction, no enterprise sales call between you and an on-demand H100. Four GPU families form the core of the lineup, from the budget A16 up to the H100:

| Plan | GPU & VRAM | Hourly price | Best for | Deploy |
| --- | --- | --- | --- | --- |
| Cloud GPU — A16 | 1× NVIDIA A16, 24 GB | from $0.18/hr | virtual workstations, graphics, budget inference | [Deploy the A16 plan](https://www.vultr.com/products/cloud-gpu/?ref=7268283) |
| Cloud GPU — L40S | 1× NVIDIA L40S, 48 GB | $1.43/hr | inference and fine-tuning for models that fit in 48 GB | [Deploy the L40S plan](https://www.vultr.com/products/cloud-gpu/?ref=7268283) |
| Cloud GPU — A100 | 1× NVIDIA A100, 80 GB | $1.98/hr | training runs that want 80 GB of HBM memory | [Deploy the A100 plan](https://www.vultr.com/products/cloud-gpu/?ref=7268283) |
| Cloud GPU — H100 | 1× NVIDIA H100, 80 GB | $2.99/hr | LLM fine-tuning, high-throughput serving | [Deploy the H100 plan](https://www.vultr.com/products/cloud-gpu/?ref=7268283) |

Larger 2× and 4× A16 configurations exist alongside the single-GPU plans, and newer Hopper and Blackwell options (H200, plus B200 on Bare Metal) have been landing on the same product page — worth a look at the live lineup before you settle. 👉 [See Vultr's full GPU lineup and current hourly rates](https://www.vultr.com/products/cloud-gpu/?ref=7268283)

Billing is hourly with a 672-hour monthly cap, which means a GPU left running around the clock never costs more than 28 days' worth, even in a 31-day month. Concretely, if an instance never turns off:

- 1× A16: about $121/month
- L40S: about $961/month
- A100 80 GB: about $1,331/month
- H100 80 GB: about $2,009/month

Deployment is self-serve. Pick a plan and region from the console or API, and the instance lands in a normal Vultr cloud environment: VPC networking, block storage you can snapshot, object storage for model weights, and the option to attach the GPU node to a Vultr Kubernetes Engine cluster. Nothing exotic to learn, and no separate workflow from the rest of your infrastructure.

## The VRAM math that actually decides your plan

VRAM is the first constraint to check, because it's the one that disqualifies a plan outright. A 70B model needs 32 GB+ of VRAM even with aggressive quantization, and a 405B model needs 250 GB+. Framework support, throughput and price only matter after the model fits.

For serving, the flagship isn't automatically the right answer. If your model fits comfortably in 48 GB, Vultr's L40S at $1.43/hr does the same job the H100 does at $2.99/hr for many workloads — at typical inference batch sizes, a large share of the H100's extra compute just sits idle. An L4, L40S or A10-class card often wins the cost-per-token fight when memory, rather than raw FLOPS, is the binding constraint.

Fine-tuning shifts the calculus. LoRA and QLoRA runs on 70B-class models are comfortable on an 80 GB A100 or H100, while full fine-tuning wants multiple GPUs. The A100 at $1.98/hr is the interesting middle option on Vultr's lineup: older than the H100, still HBM memory, a third cheaper. For most single-node fine-tuning it's the sweet spot. 👉 [Compare the L40S, A100 and H100 hourly rates on Vultr](https://www.vultr.com/pricing/?ref=7268283)

Training across multiple nodes is a different discipline entirely. Without non-blocking interconnect between nodes, gradient synchronization eats the throughput you bought extra GPUs to obtain. That's why CoreWeave builds clusters on non-blocking 3.2 Tbps InfiniBand and Lambda sells 1-Click Clusters on Quantum-2 fabric — and why spreading an eight-GPU run across unrelated marketplace hosts usually disappoints.

## Where Vultr fits, and where it doesn't

Vultr is the right call when you want specialist-style pricing on a mainstream cloud, and the wrong call in three specific situations.

On the fit side: pricing is flat and published, so there's no auction to babysit and no spot instance that vanishes mid-run. The GPU node lives in a real cloud environment with block storage, object storage, VPC and managed Kubernetes around it — model weights persist, snapshot and reattach instead of re-downloading on every boot. And with data centers in 30+ locations, proximity to your users is less of a lottery than on providers running only a handful of regions.

Three situations where another route wins. Fault-tolerant batch work where the absolute cheapest hour matters: community-cloud RTX 4090s run $0.20–$0.35/hr on spot pricing, and marketplace H100s undercut $2.99. Spiky inference with long idle gaps: a serverless endpoint that scales to zero with per-second billing beats paying a full hourly rate for a GPU that's mostly asleep. And hyperscale pretraining of 70B–405B models: multi-node interconnect is the deciding factor there, which is CoreWeave's and Lambda's territory.

On pure H100 price, Vultr's $2.99/hr lands mid-band for fixed-price providers — cheaper than CoreWeave's committed enterprise rates, pricier than the marketplace floor, in the same zone as io.net and slightly above Lambda's on-demand. You're paying a modest premium over the rock-bottom options for predictable capacity and the surrounding platform. Whether that premium is worth it depends on which of the three situations above describes you.

## Billing details that decide your real invoice

Four details matter more than headline hourly rates: granularity, the monthly cap, egress, and what happens when a job ends.

1. **Billing granularity.** RunPod and Together AI bill per second; Vast.ai and Lambda per minute; Vultr hourly. A 20-minute experiment costs roughly three times more on hourly billing, so short, iterative sessions favor the per-second providers.
2. **The monthly cap.** Vultr caps hourly billing at 672 hours — 28 days — so long months don't cost proportionally more.
3. **Egress.** Budget $0.08–$0.12/GB at the hyperscalers; $0 at providers like Hyperstack. Vultr includes a bandwidth allowance with its plans, and the exact amount is listed per plan on the pricing page, so check it if your workload moves a lot of data.
4. **Stopped vs destroyed.** On hourly billing, verify whether stopping an instance actually stops the meter — on several providers it doesn't. Destroying is the only guaranteed off switch.

Commitments change the math for steady-state workloads: Lambda's reserved pricing runs about 50% off on-demand, and CoreWeave operates on committed contracts. Vultr's on-demand rates carry no lock-in, which matters if your training queue is lumpy rather than constant.

## Frequently asked questions

### How much VRAM does a 70B model need?

32 GB is the practical floor with aggressive quantization, and 80 GB is the comfortable target for serving or fine-tuning. A 405B model needs 250 GB or more, which pushes you into multi-GPU territory regardless of provider.

### Do I need an H100 for inference?

Usually not. If the model fits in 48 GB, an L40S-class card serves it at roughly half the hourly price, and at typical batch sizes much of the H100's extra compute goes unused. Reach for the flagship when memory bandwidth or throughput actually binds.

### Why is AWS so much more expensive for the same GPU?

You're paying for the ecosystem around the chip — procurement, compliance, quota systems, integration — plus egress surcharges of $0.08–$0.12/GB. An H100 works out to about $12.30/hr there versus $2–3 at specialty clouds. If your company already runs on AWS and holds credits, that calculus changes; for everyone else it's hard to justify.

### Are decentralized GPU networks a serious option?

For fault-tolerant workloads, increasingly yes. io.net pools over 320,000 GPUs across 130+ countries and prices H100s at $2.10–$3.50/hr, consistently undercutting centralized providers. It's a younger platform with maturing tooling, so most teams point at it the workloads that can survive a node disappearing — batch jobs, distributed inference — rather than multi-day training runs.

## Picking by workload: the short version

- Learning and experimentation, lowest cost per hour wins → the marketplace route (Vast.ai) or community-cloud spot 4090s.
- Inference endpoints with unpredictable traffic → a serverless endpoint that scales to zero (RunPod, Together AI).
- Pretraining 70B–405B models across nodes → CoreWeave or Lambda, where InfiniBand comes standard.
- A flat, predictable hourly rate on a real cloud with storage, VPC and Kubernetes attached → Vultr: L40S at $1.43/hr if 48 GB covers your model, H100 at $2.99/hr when it doesn't.

Two checks before you click deploy. First, confirm the model's actual VRAM need — the jump from L40S ($1.43/hr) to H100 ($2.99/hr) is the single biggest cost lever on Vultr's lineup, and plenty of inference workloads never need to make that jump. Second, get in the habit of destroying instances the moment a job ends, because hourly billing doesn't care whether the GPU is busy. 👉 [Open Vultr's Cloud GPU pricing and pick your plan](https://www.vultr.com/products/cloud-gpu/?ref=7268283)
