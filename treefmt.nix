# treefmt-nix configuration — unified multi-language format + shell lint.
#
# Wired into `formatter.<system>` and `checks.<system>.treefmt` in flake.nix:
# `nix fmt` applies, `nix flake check` verifies.
#
# DIVERGE: python (ruff format / ruff check) is deliberately NOT run through
# treefmt. The uv-pinned ruff in `make check` is the single source of truth
# for python; a second, nixpkgs-pinned ruff here could disagree after either
# side bumps and split the gates this template exists to keep aligned.
{ ... }:
{
  projectRootFile = "flake.nix";

  # Nix — official RFC 166 style. Paired with the statix lint check.
  programs.nixfmt.enable = true;

  # Shell — 2-space indent, POSIX defaults otherwise. The gate scripts in
  # .githooks/ have no extension, so they are matched explicitly.
  programs.shfmt = {
    enable = true;
    indent_size = 2;
  };
  settings.formatter.shfmt.includes = [ ".githooks/*" ];

  # Shell — static analysis (same rigor for extensionless gate scripts).
  programs.shellcheck.enable = true;
  settings.formatter.shellcheck.includes = [ ".githooks/*" ];

  # YAML — keep single blank lines (workflow files use them for grouping).
  programs.yamlfmt = {
    enable = true;
    settings.formatter.retain_line_breaks_single = true;
  };

  # Lockfiles are tool-managed; never reformat them.
  settings.global.excludes = [ "*.lock" ];
}
