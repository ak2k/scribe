{
  description = "A new Python project (uv-managed; Nix dev shell + uv2nix build).";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs = {
        pyproject-nix.follows = "pyproject-nix";
        nixpkgs.follows = "nixpkgs";
      };
    };
    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs = {
        pyproject-nix.follows = "pyproject-nix";
        uv2nix.follows = "uv2nix";
        nixpkgs.follows = "nixpkgs";
      };
    };
    treefmt-nix = {
      url = "github:numtide/treefmt-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      pyproject-nix,
      uv2nix,
      pyproject-build-systems,
      treefmt-nix,
    }:
    let
      inherit (nixpkgs) lib;
      systems = [
        "aarch64-darwin"
        "x86_64-darwin"
        "aarch64-linux"
        "x86_64-linux"
      ];
      forAllSystems = lib.genAttrs systems;
      pkgsFor = system: nixpkgs.legacyPackages.${system};

      # Python interpreter follows pyproject.toml's `requires-python`: the
      # first PEP 440 spec is taken as the floor (handles `>=X.Y` and
      # `>=X.Y,<Z.W` shapes). Bump pyproject.toml — flake follows.
      project = pyproject-nix.lib.project.loadPyproject { projectRoot = ./.; };
      lowerBound = lib.head project.requires-python;
      pyAttr =
        "python"
        + toString (builtins.elemAt lowerBound.version.release 0)
        + toString (builtins.elemAt lowerBound.version.release 1);

      # uv.lock → Nix. Wheels preferred (faster, matches what `uv sync` resolves).
      # Switch to "sdist" if a dep ships a broken wheel and needs local build.
      workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };
      overlay = workspace.mkPyprojectOverlay { sourcePreference = "wheel"; };

      # Editable variant: project source resolved from $REPO_ROOT at shell
      # entry, not baked into the store. Source edits show up immediately;
      # uv.lock changes still require re-entering the shell.
      editableOverlay = workspace.mkEditablePyprojectOverlay { root = "$REPO_ROOT"; };

      pythonSets = forAllSystems (
        system:
        let
          pkgs = pkgsFor system;
          python = pkgs.${pyAttr};
        in
        (pkgs.callPackage pyproject-nix.build.packages { inherit python; }).overrideScope (
          lib.composeManyExtensions [
            pyproject-build-systems.overlays.wheel
            overlay
          ]
        )
      );

      # Unified formatter pipeline (see treefmt.nix for what it covers).
      treefmtEval = forAllSystems (system: treefmt-nix.lib.evalModule (pkgsFor system) ./treefmt.nix);
    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = pkgsFor system;
          python = pkgs.${pyAttr};

          # Editable pythonSet — only used by `.#pure`. Hatchling's
          # `build_editable` hook imports the `editables` package; inject it
          # as a build input here (not in the base set — production builds
          # don't need it).
          editablePythonSet = pythonSets.${system}.overrideScope (
            lib.composeManyExtensions [
              editableOverlay
              (final: prev: {
                scribe = prev.scribe.overrideAttrs (old: {
                  nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ final.resolveBuildSystem { editables = [ ]; };
                });
              })
            ]
          );
          editableVenv = editablePythonSet.mkVirtualEnv "scribe-dev-env" workspace.deps.all;

          # Arm the tracked pre-push gate via the guarded installer — a real
          # script (shellcheck-linted by checks.shellcheck, regression-tested
          # by checks.gate), executed as its own `sh` process so the user's
          # interactive shell options (errexit/pipefail/zsh nomatch) cannot
          # alter its control flow. The flake's own store copies run, and the
          # hook path argument lets the installer refuse to arm a repo whose
          # hook is not the one this dev shell ships. `|| true` only shields
          # shell entry; the installer reports its own refusals.
          installHooks = ''
            sh ${./.githooks/install} ${./.githooks/pre-push} || true
          '';
        in
        {
          # Default shell: uv-managed. Matches the inner loop (`make fix`/`check`)
          # without rebuilding the Nix venv on every dep tweak. `uv add` / `uv lock`
          # work natively; `uv sync` populates `.venv` on first entry.
          default = pkgs.mkShell {
            packages = [
              python
              pkgs.uv
              pkgs.gnumake
            ];

            # uv's bundled python-build-standalone won't link on NixOS;
            # force the Nix-managed interpreter. No-op on Darwin.
            shellHook = ''
              export UV_PYTHON=${python}/bin/python3
              export UV_PYTHON_PREFERENCE=only-system
              ${installHooks}
              echo "🐍 python: $(python3 --version)"
              echo "📦 uv:     $(uv --version)"
            '';
          };

          # Pure shell: uv2nix-built venv with the project installed editable.
          # No `.venv` needed. Source edits to src/scribe/ are live. Bumping
          # uv.lock requires exiting and re-entering so Nix re-resolves.
          # `uv lock --upgrade` works inside. Dep changes (`uv sync`, `uv add`,
          # `make install`) do NOT — the env is a read-only nix closure, so they
          # fail with a store-path permission error; use the default shell.
          pure = pkgs.mkShell {
            packages = [
              editableVenv
              pkgs.uv
              pkgs.gnumake
            ];
            env = {
              UV_NO_SYNC = "1";
              UV_PYTHON = "${editableVenv}/bin/python";
              UV_PYTHON_DOWNLOADS = "never";
              # `uv run` (incl. via make) uses the nix-built closure as the
              # project venv — never a stale `.venv` or an inherited
              # relocation path from the user's shell.
              UV_PROJECT_ENVIRONMENT = "${editableVenv}";
            };
            shellHook = ''
              unset PYTHONPATH
              export REPO_ROOT=$(git rev-parse --show-toplevel)
              ${installHooks}
              echo "🐍 python: $(python --version) (nix-built, editable)"
              echo "📦 uv:     $(uv --version)"
            '';
          };
        }
      );

      # `nix build .#default` produces a runtime venv (project + deps in
      # `[project.dependencies]`). `.#dev` adds `[dependency-groups].dev`
      # (basedpyright, ruff, pytest, ...). Both are Nix-built — no uv at
      # build time — suitable for nixpkgs PRs / NixOS modules / consumers.
      packages = forAllSystems (system: {
        default = pythonSets.${system}.mkVirtualEnv "scribe-env" workspace.deps.default;
        dev = pythonSets.${system}.mkVirtualEnv "scribe-dev-env" workspace.deps.all;
      });

      # `nix fmt` runs the unified treefmt pipeline (nix / shell / yaml —
      # python is deliberately owned by the uv-pinned ruff in `make check`;
      # see treefmt.nix). Paired with checks.${system}.treefmt below.
      formatter = forAllSystems (system: treefmtEval.${system}.config.build.wrapper);

      # `nix flake check` runs these. `runCommand … touch $out` is the
      # idiomatic pass/fail pattern: tool exits non-zero → derivation fails.
      checks = forAllSystems (
        system:
        let
          pkgs = pkgsFor system;
          # The dev venv (same derivation `nix build .#dev` produces) runs the
          # suite inside the uv2nix closure — what makes `nix flake check` a
          # faithful gate: it catches native-linking failures the wheels-based
          # raw-uv path hides. NB this does NOT typecheck; only `make check`
          # runs basedpyright (the pre-push hook runs both).
          testVenv = self.packages.${system}.dev;
        in
        {
          statix = pkgs.runCommand "check-statix" { nativeBuildInputs = [ pkgs.statix ]; } ''
            statix check ${./flake.nix}
            touch $out
          '';
          # Formatting (nix/shell/yaml) + shellcheck over the whole tree —
          # the check twin of `nix fmt`. Python format/lint is deliberately
          # NOT here (see treefmt.nix): `make check`'s uv-pinned ruff owns it.
          treefmt = treefmtEval.${system}.config.build.check self;
          # Hermetic regression matrix for the hook + installer (make/nix are
          # PATH-shimmed inside the script). Every scenario pins a behavior a
          # review round proved breakable.
          gate =
            pkgs.runCommand "check-gate"
              {
                nativeBuildInputs = [
                  pkgs.git
                  pkgs.bash
                ];
              }
              ''
                cp -r ${./.} repo && chmod -R +w repo
                # The Linux sandbox has no /usr/bin/env: point the scripts'
                # shebangs at store paths so they exec the same way CI's
                # unsandboxed uv-path run execs them.
                patchShebangs repo/.githooks repo/scripts
                cd repo && bash ./scripts/check-gate.sh
                touch $out
              '';
          # The test suite run against `src/` inside the closure. `PYTHONPATH=src`
          # shadows the installed copy so coverage/fixtures resolve to the tree.
          # `${./.}` is the flake source — git-TRACKED files only; a new test
          # must be `git add`ed before this check can see it (matching CI).
          # NB darwin nix defaults to `sandbox = false`, so this check is
          # stricter on Linux CI than locally — a test that touches the network
          # or reads /etc can pass here and fail there.
          pytest = pkgs.runCommand "check-pytest" { nativeBuildInputs = [ testVenv ]; } ''
            cp -r ${./.} work && chmod -R +w work && cd work
            export HOME="$TMPDIR"
            export PYTHONPATH="$PWD/src"
            # The Linux sandbox has no system CA bundle; tests that construct an
            # httpx client (SSL context init, no network) need a cert file.
            export SSL_CERT_FILE="${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
            # --no-cov: `make check` owns the coverage gate; this check is
            # scoped to what only it can catch (native linking, lockfile
            # drift). The flag needs pytest-cov in the dev deps to parse, and
            # this run inherits everything else in pyproject's addopts — a
            # future marker filter (e.g. -m 'not live') applies here too.
            pytest --no-cov -p no:cacheprovider
            touch $out
          '';
        }
      );
    };
}
