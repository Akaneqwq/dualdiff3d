from argparse import SUPPRESS

from dualdiff.hub import DEFAULT_MODEL_FILENAME, DEFAULT_MODEL_ID


RELIABLE_PRESETS = {
    "normal": {
        "n_fix_views": 2, "dualdiff_timestep": 199, "k_repair": 3,
        "alpha_var": 500.0, "beta_diff": 10.0, "gamma_proj": 10.0,
        "lambda_conf": (0.2, 0.3, 0.5), "conf_threshold": 0.6,
        "repair_ratio": 0.3, "rollback_patience": 3,
        "min_interp_alpha": 0.15, "max_interp_alpha": 0.5,
        "max_extrap_alpha": 0.15, "min_pose_dist": 0.03,
        "max_pose_dist": 0.75,
        "cproj_warmup_until": 10000, "cproj_pose_dist_thresh": 0.75,
        "rollback_eval_interval": 500, "profile_fix": False,
        "novel_pool_max_size": 0, "novel_pool_dedup_pose_dist": 0.03,
        "alpha_min": 1.0 / 255.0, "novel_loss_weight": 0.3,
    },
    "fast": {"k_repair": 1, "n_fix_views": 1, "repair_ratio": 0.2,
             "conf_threshold": 0.55, "rollback_eval_interval": 1000},
    "strict": {"k_repair": 4, "alpha_var": 700.0, "beta_diff": 15.0,
               "gamma_proj": 15.0, "conf_threshold": 0.7,
               "repair_ratio": 0.2, "cproj_pose_dist_thresh": 0.5},
    "debug": {"k_repair": 2, "n_fix_views": 1,
              "rollback_eval_interval": 250, "profile_fix": True},
}


def add_reliable_arguments(parser):
    parser.add_argument("--fix_iterations", nargs="+", type=int, default=list(range(500, 30000, 2000)))
    parser.add_argument("--fix", action="store_true")
    parser.add_argument("--disable_lpips", action="store_true", help="Disable validation LPIPS; rollback uses PSNR only.")
    parser.add_argument("--reliable_preset", choices=tuple(RELIABLE_PRESETS), default="normal")
    parser.add_argument("--n_fix_views", type=int, default=SUPPRESS)
    parser.add_argument("--dualdiff_ckpt", type=str, default=None,
                        help="Local checkpoint path; overrides Hugging Face download.")
    parser.add_argument("--dualdiff_model_id", type=str, default=None,
                        help=f"Hugging Face model repository (default: {DEFAULT_MODEL_ID}).")
    parser.add_argument("--dualdiff_model_filename", type=str, default=None,
                        help=f"Checkpoint filename on the Hub (default: {DEFAULT_MODEL_FILENAME}).")
    parser.add_argument("--dualdiff_model_revision", type=str, default=None,
                        help="Optional Hugging Face branch, tag, or commit.")
    parser.add_argument("--dualdiff_cache_dir", type=str, default=None,
                        help="Optional Hugging Face cache directory.")
    parser.add_argument("--dualdiff_local_files_only", action="store_true",
                        help="Use only an already-cached Hugging Face checkpoint.")
    parser.add_argument("--dualdiff_timestep", type=int, default=SUPPRESS)
    parser.add_argument("--k_repair", type=int, default=SUPPRESS)
    parser.add_argument("--alpha_var", type=float, default=SUPPRESS)
    parser.add_argument("--beta_diff", type=float, default=SUPPRESS)
    parser.add_argument("--gamma_proj", type=float, default=SUPPRESS)
    parser.add_argument("--lambda_conf", nargs=3, type=float, default=SUPPRESS)
    parser.add_argument("--conf_threshold", type=float, default=SUPPRESS)
    parser.add_argument("--repair_ratio", type=float, default=SUPPRESS)
    parser.add_argument("--rollback_patience", type=int, default=SUPPRESS)
    parser.add_argument("--min_interp_alpha", type=float, default=SUPPRESS)
    parser.add_argument("--max_interp_alpha", type=float, default=SUPPRESS)
    parser.add_argument("--max_extrap_alpha", type=float, default=SUPPRESS)
    parser.add_argument("--min_pose_dist", type=float, default=SUPPRESS)
    parser.add_argument("--max_pose_dist", type=float, default=SUPPRESS)
    parser.add_argument("--cproj_warmup_until", type=int, default=SUPPRESS)
    parser.add_argument("--cproj_pose_dist_thresh", type=float, default=SUPPRESS)
    parser.add_argument("--rollback_eval_interval", type=int, default=SUPPRESS)
    parser.add_argument("--profile_fix", action="store_true", default=SUPPRESS)
    parser.add_argument("--novel_pool_max_size", type=int, default=SUPPRESS)
    parser.add_argument("--novel_pool_dedup_pose_dist", type=float, default=SUPPRESS)
    parser.add_argument("--alpha_min", type=float, default=SUPPRESS,
                        help="Minimum rasterized alpha for valid reprojection support (default: 1/255).")
    parser.add_argument("--novel_loss_weight", type=float, default=SUPPRESS,
                        help="Weight applied to confidence-masked novel-view loss (default: 0.3).")


def apply_reliable_preset(args):
    values = dict(RELIABLE_PRESETS["normal"])
    values.update(RELIABLE_PRESETS[args.reliable_preset])
    for key, value in values.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    if args.n_fix_views < 1:
        raise ValueError("--n_fix_views must be >= 1")
    if args.rollback_eval_interval < 1:
        raise ValueError("--rollback_eval_interval must be >= 1")
    for name in ("repair_ratio", "conf_threshold"):
        if not 0.0 <= getattr(args, name) <= 1.0:
            raise ValueError(f"--{name} must be in [0, 1]")
    if args.min_pose_dist < 0.0 or args.max_pose_dist < args.min_pose_dist:
        raise ValueError("--max_pose_dist must be >= --min_pose_dist >= 0")
    if args.max_interp_alpha < args.min_interp_alpha:
        raise ValueError("--max_interp_alpha must be >= --min_interp_alpha")
    if args.novel_pool_max_size < 0:
        raise ValueError("--novel_pool_max_size must be >= 0")
    if args.novel_pool_dedup_pose_dist < 0.0:
        raise ValueError("--novel_pool_dedup_pose_dist must be >= 0")
    return args
