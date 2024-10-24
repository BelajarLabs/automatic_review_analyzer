# To learn more about how to use Nix to configure your environment
# see: https://developers.google.com/idx/guides/customize-idx-env
{ pkgs, ... }: {

  # Which nixpkgs channel to use.
  channel = "unstable"; # or "stable-24.05"

  # Use https://search.nixos.org/packages to find packages
  packages = with pkgs;[
    cargo
    clippy
    rustc
    rustfmt
    stdenv.cc
  ];

  # Sets environment variables in the workspace
  env = {
    RUST_SRC_PATH = "${pkgs.rustPlatform.rustLibSrc}";
  };
  
  # Search for the extensions you want on https://open-vsx.org/ and use "publisher.id"
  idx.extensions = [
    "rust-lang.rust-analyzer"
    "tamasfe.even-better-toml"
    "serayuzgur.crates"
    "vadimcn.vscode-lldb"
  ];
}