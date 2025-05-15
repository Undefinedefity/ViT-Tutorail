from omegaconf import DictConfig, OmegaConf
import hydra
import os
# @hydra.main(version_base=None, config_path=".", config_name="config")
# def my_app(cfg: DictConfig):
#     # assert cfg.node.loompa == 10          # attribute style access
#     # assert cfg["node"]["loompa"] == 10    # dictionary style access

#     # assert cfg.node.zippity == 10         # Value interpolation
#     # assert isinstance(cfg.node.zippity, int)  # Value interpolation type
#     # assert cfg.node.do == "oompa 10"      # string interpolation
#     print(OmegaConf.to_yaml(cfg))

#     print("cfg.node.loompa: ",cfg.node.loompa)
#     print("cfg['node']['loompa']: ",cfg["node"]["loompa"])
#     print("cfg.node.zippity: ",cfg.node.zippity)
#     print("cfg.node.zippity type: ",type(cfg.node.zippity))
#     print("cfg.node.do: ",cfg.node.do)
#     print("cfg.node.do type: ",type(cfg.node.do))
#     cfg.node.waldo                        # raises an exception



# -------------------------------------------------

@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    print(f"Working directory : {os.getcwd()}")
    print(f"Output directory  : {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")


if __name__ == "__main__":
    my_app()