import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import numpy as np
import os
import sys

from binferno.get_augfeature import get_ContrastiveLearningDataset, GetCorgiContrastiveDataset
from binferno.utils import get_length

from binferno.models.mlp import EmbeddingNet
from binferno.simclr import SimCLR

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
# torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True


def train_CLmodel(logger, device=None, num_threads=10, output_path=None, data=None,
                n_views=6, notuseaug0=False, cov_meannormalize=False, cov_minmaxnormalize=False,
                cov_standardization=False, addvars=False, vars_sqrt=False, corgi_l2_normalize=True, merge_kmer_corgi=False,
                batch_size=256, epochs=100, lr=1e-3, weight_decay=1e-6, dropout_value=0.2, fp16_precision=False,
                emb_szs=2048, n_layers=3, out_dim=128, out_dim_forcov=128, emb_szs_forcov=2048, n_layers_forcov=3,
                pretrain_kmer_model_path="no", not_load_kmermetric_state=False, finetunepretrainmodel=False, early_stop=False,
                finetunelr_ratio=0.1, covmodelepochs=50, covmodel_notl2normalize=False, contig_len=1000, workers=4, 
                add_model_for_coverage=False, plastid=False, kmer_model_path="empty", pretrain_coveragemodel=False,
                args_dict=None):
    """
    Train the Contrastive Learning model.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        torch.set_num_threads(num_threads)

    logger.info("Generate features for the data.")

    if os.path.exists(output_path + "/embeddings.tsv"):
        logger.info(
            "The embeddings file has been generated before, please check the output commands."
        )
        sys.exit()

    dataset, namelist = GetCorgiContrastiveDataset(
        data_path=data,
        n_views=n_views,
        device=device,
        cov_meannormalize=cov_meannormalize,
        cov_minmaxnormalize=cov_minmaxnormalize,
        cov_standardization=cov_standardization,
        addvars=addvars,
        vars_sqrt=vars_sqrt,
        corgi_l2_normalize=corgi_l2_normalize,
        merge_kmer_corgi=merge_kmer_corgi
    )

    contig_file = data + "/aug0/sequences_aug0.fasta"
    lengths = get_length(contig_file)
    length_weight = []
    for seq_id in namelist:
        length_weight.append(lengths[seq_id])

    if notuseaug0:
        n_views = n_views - 1
        train_dataset = torch.utils.data.TensorDataset(
            *[
                dataset[i + 1][np.array(length_weight) >= contig_len]
                for i in range(n_views)
            ]
        )
    else:
        # train_dataset = torch.utils.data.TensorDataset(*dataset)
        train_dataset = torch.utils.data.TensorDataset(
            *[
                dataset[i][np.array(length_weight) >= contig_len]
                for i in range(n_views)
            ]
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
    )

    # Set embedder model.
    if not add_model_for_coverage:
        ps = [dropout_value] * (n_layers - 1)
        actn = nn.LeakyReLU()

        emb_szs_list = [emb_szs] * n_layers

        model = EmbeddingNet(
            in_sz=len(dataset[0][0]),
            out_sz=out_dim,
            emb_szs=emb_szs_list,
            ps=ps,
            use_bn=True,
            actn=actn,
        )

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=0, last_epoch=-1
        )

        simclr = SimCLR(
            model=model, optimizer=optimizer, scheduler=scheduler, args_dict=args_dict, output_path=output_path, device=device, plastid=plastid
        )
        simclr.train(train_loader, dataset, namelist)

    else:
        if pretrain_kmer_model_path != "no":
            config_file = (
                os.path.dirname(pretrain_kmer_model_path)
                + "/kmerMetric_config.yaml"
            )

            from ruamel.yaml import YAML
            from pathlib import Path

            yaml = YAML(typ="safe")

            cnf = yaml.load(Path(config_file))

            ps = [cnf["dropout_value"]] * (len(cnf["emb_szs"]) - 1)
            actn = nn.LeakyReLU()

            insize = 136

            kmerMetric_model = EmbeddingNet(
                in_sz=insize,
                out_sz=cnf["embedding_dim"],
                emb_szs=cnf["emb_szs"],
                ps=ps,
                use_bn=True,
                actn=actn,
            )

            kmerMetric_model = kmerMetric_model.to(device)

            if not not_load_kmermetric_state:
                kmerMetric_model.load_state_dict(
                    torch.load(pretrain_kmer_model_path, map_location=device)
                )

            cov_dim = len(dataset[0][0]) - insize
            input_size = out_dim_forcov + 128
            print("cov_dim:\t" + str(cov_dim) + "\n")
    
            emb_szs_list = [emb_szs_forcov] * n_layers_forcov

            cov_model = EmbeddingNet(
                in_sz=cov_dim,
                out_sz=out_dim_forcov,
                emb_szs=emb_szs_list,
                ps=[dropout_value] * (n_layers_forcov - 1),
                use_bn=True,
                actn=nn.LeakyReLU(),
            )

            from binferno.models.mlp2 import EmbeddingNet as EmbeddingNet2

            emb_szs_list = [emb_szs] * n_layers

            model = EmbeddingNet2(
                in_sz=input_size,
                out_sz=out_dim,
                emb_szs=emb_szs_list,
                ps=[dropout_value] * (n_layers - 1),
                use_bn=True,
                actn=nn.LeakyReLU(),
                cov_model=cov_model,
                pretrained_model=kmerMetric_model,
                covmodel_notl2normalize=covmodel_notl2normalize,
            )

            if finetunepretrainmodel:
                pretrained_model_paraname = []
                for name, param in model.pretrained_model.named_parameters():
                    # print(name)
                    pretrained_model_paraname.append("pretrained_model." + name)
                print(pretrained_model_paraname)

                params_lx = [
                    param
                    for name, param in model.named_parameters()
                    if name not in pretrained_model_paraname
                ]
                
                optimizer = torch.optim.AdamW(
                    [
                        {"params": params_lx},
                        {
                            "params": model.pretrained_model.parameters(),
                            "lr": lr * finetunelr_ratio,
                        },
                    ],
                    lr=lr,
                    weight_decay=weight_decay,
                )

            else:
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=lr, weight_decay=weight_decay
                )

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=0, last_epoch=-1
            )

            simclr = SimCLR(
                model=model, optimizer=optimizer, scheduler=scheduler, args_dict=args_dict, output_path=output_path, device=device, plastid=plastid
            )
            simclr.train_addpretrain(train_loader, dataset, namelist, fp16_precision=fp16_precision, device=device, 
                                     epochs=epochs, notuse_scheduler=False, earlystop=early_stop, output_path=output_path)

        else:
            if kmer_model_path == "empty":
                if plastid:
                    if not merge_kmer_corgi:
                        cov_dim = len(dataset[0][0]) - 538
                        input_size = out_dim_forcov + 538
                    else:
                        cov_dim = len(dataset[0][0]) - 538 - 136
                        input_size = out_dim_forcov + 538 + 136
                else:
                    if not merge_kmer_corgi:
                        cov_dim = len(dataset[0][0]) - 1945
                        input_size = out_dim_forcov + 1945
                        if cov_dim < 0:
                            cov_dim = len(dataset[0][0]) - 538
                            input_size = out_dim_forcov + 538
                    else:
                        # Sometimes it errornously got classified here.
                        cov_dim = len(dataset[0][0]) - 1945 - 136
                        input_size = out_dim_forcov + 1945 + 136
                        if cov_dim < 0:
                            cov_dim = len(dataset[0][0]) - 538 - 136
                            input_size = out_dim_forcov + 538 + 136

            else:
                cov_dim = len(dataset[0][0]) - 128
                input_size = out_dim_forcov + 128
                print("cov_dim:\t" + str(cov_dim) + "\n")

            emb_szs_list = [emb_szs_forcov] * n_layers_forcov

            cov_model = EmbeddingNet(
                in_sz=cov_dim,
                out_sz=out_dim_forcov,
                emb_szs=emb_szs_list,
                ps=[dropout_value] * (n_layers_forcov - 1),
                use_bn=True,
                actn=nn.LeakyReLU(),
            )

            if pretrain_coveragemodel:
                print("pretrain_coveragemodel!")
                print(cov_model.state_dict())
                print(len(dataset[0][0][128:]))
                optimizer = torch.optim.AdamW(
                    cov_model.parameters(), lr=lr, weight_decay=weight_decay
                )

                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=covmodelepochs, eta_min=0, last_epoch=-1
                )

                simclr = SimCLR(
                    model=cov_model, optimizer=optimizer, scheduler=scheduler, args_dict=args_dict, output_path=output_path, device=device, plastid=plastid
                )
                simclr.covmodeltrain(train_loader)
                print(cov_model.state_dict())
                print(len(dataset[0][0][128:]))

            from binferno.models.mlp2 import EmbeddingNet as EmbeddingNet2

            emb_szs_list = [emb_szs] * n_layers

            model = EmbeddingNet2(
                in_sz=input_size,
                out_sz=out_dim,
                emb_szs=emb_szs_list,
                ps=[dropout_value] * (n_layers - 1),
                use_bn=True,
                actn=nn.LeakyReLU(),
                cov_model=cov_model,
                covmodel_notl2normalize=covmodel_notl2normalize,
            )
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=0, last_epoch=-1
            )

            simclr = SimCLR(
                model=model, optimizer=optimizer, scheduler=scheduler, args_dict=args_dict, output_path=output_path, device=device, plastid=plastid
            )
            simclr.train_addpretrain(train_loader, dataset, namelist, fp16_precision=fp16_precision, device=device, merge_kmer_corgi=merge_kmer_corgi,
                                     epochs=epochs, notuse_scheduler=False, earlystop=early_stop, output_path=output_path, kmer_dim=(len(dataset[0][0])-cov_dim))

    logger.info("Finish training.")
