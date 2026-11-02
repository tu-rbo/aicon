// The module 'vscode' contains the VS Code extensibility API
// Import the module and reference it with the alias vscode in your code below
import * as vscode from 'vscode';
import * as path from 'path';
import * as os from 'os';
import { constants, existsSync } from "node:fs";
import { access, mkdir } from "node:fs/promises";
import { spawn } from "node:child_process";

import * as semver from "semver";

import { PythonExtension } from "@vscode/python-extension";
import {
    LanguageClient,
    LanguageClientOptions,
    ServerOptions,
    State,
    Trace,
} from "vscode-languageclient/node";
import { registerSpecVisualizerCommand } from "./specVisualizer";
let client: LanguageClient | undefined;
let clientStarting = false;
let restartRequested = false;
let python: PythonExtension | undefined;
let logger: vscode.LogOutputChannel;
let traceLogger: vscode.LogOutputChannel;
let extensionContext: vscode.ExtensionContext;
let currentProjectRoot: string | undefined;

type TraceSetting = "off" | "messages" | "verbose";

const MIN_PYTHON = semver.parse("3.10.0");
const DEFAULT_SERVER_GIT_REPOSITORY = "https://git.tu-berlin.de/ole_l/aiconls";
const INSTALLED_SERVER_DIR = "aiconls-server";
const SERVER_ENTRYPOINT = "linter.py";
const SERVER_PACKAGE = "aiconls";
// A copy of the server inside the opened repository (as in the AICON repository), used before an installed one.
const BUNDLED_SERVER_PATH = path.join("tools", "aiconls", SERVER_ENTRYPOINT);
const PROJECT_SPEC_FILE = "spec.yaml";
const REQUIRED_SERVER_MODULES = ["pygls", "yaml", "lsprotocol", "libcst"] as const;
const GENERATE_FILES_SERVER_COMMAND = "aiconls.generateFiles";
const GENERATE_EXPERIMENT_SPEC_SERVER_COMMAND = "aiconls.generateExperimentSpecification";
const FIX_ALL_SPEC_MISMATCHES_SERVER_COMMAND = "aiconls.fixAllSpecMismatches";
const RUN_DYNAMIC_CHECK_SERVER_COMMAND = "aiconls.runDynamicCheck";
const GENERATE_FILES_EXTENSION_COMMAND = "aiconls.runGenerateFiles";
const GENERATE_EXPERIMENT_SPEC_EXTENSION_COMMAND = "aiconls.runGenerateExperimentSpecification";
const FIX_ALL_SPEC_MISMATCHES_EXTENSION_COMMAND = "aiconls.runFixAllSpecMismatches";
const RUN_DYNAMIC_PARSING_EXTENSION_COMMAND = "aiconls.runDynamicParsing";
const VISUALIZE_SPEC_EXTENSION_COMMAND = "aiconls.visualizeSpec";
const SET_PROJECT_ROOT_EXTENSION_COMMAND = "aiconls.setProjectRootToCurrentFolder";
const VISUALIZE_SPEC_STATUS_BAR_TEXT = "Visualize spec.yaml";
const VISUALIZE_SPEC_CODELENS_TITLE = "Visualize spec.yaml";

export async function activate(context: vscode.ExtensionContext) {
    extensionContext = context;

	logger = vscode.window.createOutputChannel('aiconls', { log: true });
    traceLogger = vscode.window.createOutputChannel('aiconls trace', { log: true });
    context.subscriptions.push(logger, traceLogger);

	logger.info("Activated VS-Aiconls");

	python = await getPythonExtension();
    registerSpecVisualizerCommand(context, logger);
    registerSpecVisualizerStatusBar(context);
    registerSpecVisualizerCodeLens(context);

    if (python) {
        // Restart the language server if the user switches Python envs...
        context.subscriptions.push(
            python.environments.onDidChangeActiveEnvironmentPath(async () => {
                logger.info('python env modified, restarting server...');
                await startLangServer();
            })
        );
    } else {
        const message = [
            "Unable to activate AICON language features because the Python extension API is unavailable.",
            "Install/enable ms-python.python, then reload the window."
        ].join(" ");
        logger.error(message);
        void vscode.window.showErrorMessage(message);
    }

    // Restart language server command
    context.subscriptions.push(
        vscode.commands.registerCommand("aiconls.restart", async () => {
            logger.info('restarting server...');
            await startLangServer();
            }));
    context.subscriptions.push(
        vscode.commands.registerCommand("aiconls.installServer", async () => {
            logger.info('installing/updating language server from git...');
            const resource = getPythonEnvironmentResource();
            const pythonCommand = await getPythonCommand(resource);
            if (!pythonCommand) {
                return;
            }
            await installOrUpdateServerFromGit(pythonCommand, getInstalledServerRoot());
            await startLangServer();
        }));

    // Generate code from spec
    context.subscriptions.push(
        vscode.commands.registerCommand(GENERATE_FILES_EXTENSION_COMMAND, async () => {
            logger.info('generating aicon files...');
            await executeLanguageServerCommand(GENERATE_FILES_SERVER_COMMAND);
            }));
    context.subscriptions.push(
        vscode.commands.registerCommand(GENERATE_EXPERIMENT_SPEC_EXTENSION_COMMAND, async () => {
            logger.info('generating aicon experiment specification...');
            await executeLanguageServerCommand(GENERATE_EXPERIMENT_SPEC_SERVER_COMMAND);
            }));
    context.subscriptions.push(
        vscode.commands.registerCommand(FIX_ALL_SPEC_MISMATCHES_EXTENSION_COMMAND, async () => {
            logger.info("fixing all aicon spec mismatches...");
            await executeLanguageServerCommand(FIX_ALL_SPEC_MISMATCHES_SERVER_COMMAND);
        }));
    context.subscriptions.push(
        vscode.commands.registerCommand(RUN_DYNAMIC_PARSING_EXTENSION_COMMAND, async () => {
            logger.info("running dynamic parsing check...");
            await executeLanguageServerCommand(RUN_DYNAMIC_CHECK_SERVER_COMMAND);
        }));
    context.subscriptions.push(
        vscode.commands.registerCommand(SET_PROJECT_ROOT_EXTENSION_COMMAND, async (uri?: vscode.Uri) => {
            await setProjectRootToFolderOf(uri);
        }));

    // Without a configured project root, the project is the one of the active file: restart the language server
    // when a file of another project (another spec.yaml) becomes active.
    context.subscriptions.push(
        vscode.window.onDidChangeActiveTextEditor(async () => {
            if (!client || getConfiguredProjectRoot()) {
                return;
            }
            const projectRoot = getAiconProjectRoot();
            if (projectRoot !== currentProjectRoot) {
                logger.info(`project changed to '${projectRoot}', restarting server...`);
                await startLangServer();
            }
        })
    );

    // Start the language server once the user opens the first text document...
    context.subscriptions.push(
        vscode.workspace.onDidOpenTextDocument(
            async () => {
                if (!client) {
                    await startLangServer();
                }
            }
        )
    );
}

function registerSpecVisualizerStatusBar(context: vscode.ExtensionContext): void {
    const statusBarItem = vscode.window.createStatusBarItem(
        vscode.StatusBarAlignment.Right,
        1000
    );
    statusBarItem.text = VISUALIZE_SPEC_STATUS_BAR_TEXT;
    statusBarItem.command = VISUALIZE_SPEC_EXTENSION_COMMAND;
    statusBarItem.tooltip = "Open the visualizer for spec.yaml.";

    const updateStatusBarVisibility = (editor: vscode.TextEditor | undefined): void => {
        if (isSpecYamlEditor(editor)) {
            statusBarItem.show();
            return;
        }
        statusBarItem.hide();
    };

    context.subscriptions.push(
        statusBarItem,
        vscode.window.onDidChangeActiveTextEditor((editor) => {
            updateStatusBarVisibility(editor);
        })
    );

    updateStatusBarVisibility(vscode.window.activeTextEditor);
}

function registerSpecVisualizerCodeLens(context: vscode.ExtensionContext): void {
    const provider: vscode.CodeLensProvider = {
        provideCodeLenses(document: vscode.TextDocument): vscode.CodeLens[] {
            if (!isSpecYamlDocument(document)) {
                return [];
            }

            const lensPosition = new vscode.Position(0, 0);
            const lensRange = new vscode.Range(lensPosition, lensPosition);
            const command: vscode.Command = {
                title: VISUALIZE_SPEC_CODELENS_TITLE,
                command: VISUALIZE_SPEC_EXTENSION_COMMAND,
                tooltip: "Open the visualizer for this spec.yaml file.",
            };
            return [new vscode.CodeLens(lensRange, command)];
        },
    };

    context.subscriptions.push(
        vscode.languages.registerCodeLensProvider(
            { language: "yaml", scheme: "file" },
            provider
        )
    );
}

function isSpecYamlEditor(editor: vscode.TextEditor | undefined): boolean {
    if (!editor) {
        return false;
    }
    return isSpecYamlDocument(editor.document);
}

function isSpecYamlDocument(document: vscode.TextDocument): boolean {
    return path.basename(document.uri.fsPath).toLowerCase() === "spec.yaml";
}

// This method is called when your extension is deactivated
export function deactivate() {
    return stopLangServer();
}

async function startLangServer() {
    if(clientStarting) {
	    restartRequested = true;
        logger.info("Server already starting, queueing restart");
	    return;
    }

    clientStarting = true;
    restartRequested = false;

    try {
        if (client) {
            logger.info("Stopping old instance");
            await stopLangServer();
        }

	    const cwd = getCwd();
	    if(!cwd) {
		    return;
	    }

        const resource = getPythonEnvironmentResource();
        logger.info(`python env resource: '${resource?.toString() ?? "workspace-default"}'`);

        const pythonCommand = await getPythonCommand(resource);
        if (!pythonCommand) {
            return;
        }
        const serverPath = await getServerPath(pythonCommand);
        if (!serverPath) {
            return;
        }
        if (!(await ensureServerDependencies(pythonCommand, serverPath, cwd))) {
            return;
        }

        logger.info(`cwd: '${cwd}'`);
        logger.info(`server: '${serverPath}'`);
        logger.info(`python: ${pythonCommand.join(" ")}`);
        const serverLogFile = getServerLogFile();
        const aiconProjectRoot = getAiconProjectRoot();
        currentProjectRoot = aiconProjectRoot;
        logger.info(`project root: '${aiconProjectRoot}'`);
        logger.info(`server log file: '${serverLogFile}'`);
        const serverOptions: ServerOptions = {
            command: pythonCommand[0],
            args: [...pythonCommand.slice(1), serverPath],
            options: {
                cwd,
                env: {
                    ...process.env,
                    AICONLS_LOG_FILE: serverLogFile,
                    AICONLS_PROJECT_ROOT: aiconProjectRoot,
                    ...getServerDebugEnvironment(),
                },
            },
        };

        client = new LanguageClient('aiconls', serverOptions, getClientOptions());
        const traceLevel = getTraceLevel();
        client.setTrace(traceLevel);
        logger.info(`trace level: ${Trace[traceLevel]}`);
        const results = await Promise.allSettled([client.start()]);

        for (const result of results) {
            if (result.status === "rejected") {
                logger.error(`There was a error starting the server: ${result.reason}`);
            }
        }
    } finally {
        clientStarting = false;

        if (restartRequested) {
            restartRequested = false;
            logger.info("Processing queued restart");
            void startLangServer();
        }
    }



}

async function stopLangServer(): Promise<void> {
    if (!client) {
        return;
    }

    if (client.state === State.Running) {
        await client.stop();
    }

    client.dispose();
    client = undefined;
}

async function executeLanguageServerCommand(command: string): Promise<void> {
    await startLangServer();
    if (!client || client.state !== State.Running) {
        const message = `Language server is not running, cannot execute '${command}'.`;
        logger.error(message);
        void vscode.window.showErrorMessage(message);
        return;
    }

    await client.sendRequest("workspace/executeCommand", {
        command,
        arguments: [],
    });
}

function getClientOptions(): LanguageClientOptions {
    const options: LanguageClientOptions= {
        documentSelector: [{scheme: "file", language: "python"}, {scheme: "file", language: "yaml"}],
        outputChannel: logger,
        traceOutputChannel: traceLogger,
        connectionOptions: {
            maxRestartCount: 0 // don't restart on server failure.
        },
    };
    logger.info('client options configured');
    return options;
}

function getTraceLevel(): Trace {
    const traceSetting = vscode.workspace
        .getConfiguration('aiconsls')
        .get<TraceSetting>('trace.server', 'verbose');

    switch (traceSetting) {
        case 'messages':
            return Trace.Messages;
        case 'verbose':
            return Trace.Verbose;
        default:
            return Trace.Off;
    }
}

function getConfiguredProjectRoot(): string | undefined {
    const aiconRoot = vscode.workspace.getConfiguration("aiconls").get<string>("project.root");
    return aiconRoot ? aiconRoot.replace("${workspace}", getCwd()) : undefined;
}

/**
 * The AICON project root: the configured one, else the folder of the nearest spec.yaml above the active file (e.g.
 * one example of a repository with several), else the current one or the workspace.
 */
function getAiconProjectRoot(): string {
    const configuredRoot = getConfiguredProjectRoot();
    if (configuredRoot) {
        return configuredRoot;
    }
    const specRoot = findSpecRoot(vscode.window.activeTextEditor?.document.uri);
    if (specRoot) {
        return specRoot;
    }
    return currentProjectRoot ?? getCwd();
}

function findSpecRoot(uri: vscode.Uri | undefined): string | undefined {
    if (!uri || uri.scheme !== "file") {
        return;
    }
    const workspaceFolder = vscode.workspace.getWorkspaceFolder(uri)?.uri.fsPath;
    if (!workspaceFolder) {
        return;
    }
    let directory = path.dirname(uri.fsPath);
    while (true) {
        if (existsSync(path.join(directory, PROJECT_SPEC_FILE))) {
            return directory;
        }
        const parent = path.dirname(directory);
        if (directory === workspaceFolder || parent === directory) {
            return;
        }
        directory = parent;
    }
}

/**
 * Set the AICON project root to the folder of the given resource and restart the server.
 *
 * Without a resource (Command Palette), the folder of the active editor's file is used.
 * From the Explorer context menu, a folder is used as is and a file is replaced by its folder.
 */
async function setProjectRootToFolderOf(uri?: vscode.Uri): Promise<void> {
    const resource = uri ?? vscode.window.activeTextEditor?.document.uri;
    if (!resource || resource.scheme !== "file") {
        const message = "No file is open, cannot set the AICON project root.";
        logger.error(message);
        void vscode.window.showErrorMessage(message);
        return;
    }

    const folder = (await isDirectory(resource)) ? resource.fsPath : path.dirname(resource.fsPath);

    // Keep the setting portable when the folder is inside the workspace.
    const workspaceRoot = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
    let projectRoot = folder;
    if (workspaceRoot) {
        const relative = path.relative(workspaceRoot, folder);
        if (!relative.startsWith("..") && !path.isAbsolute(relative)) {
            projectRoot = relative ? path.posix.join("${workspace}", ...relative.split(path.sep)) : "${workspace}";
        }
    }

    await vscode.workspace
        .getConfiguration("aiconls")
        .update("project.root", projectRoot, vscode.ConfigurationTarget.Workspace);
    logger.info(`AICON project root set to '${projectRoot}' (${folder}), restarting server...`);
    void vscode.window.showInformationMessage(`AICON project root set to '${folder}'.`);
    await startLangServer();
}

async function isDirectory(uri: vscode.Uri): Promise<boolean> {
    try {
        return ((await vscode.workspace.fs.stat(uri)).type & vscode.FileType.Directory) !== 0;
    } catch {
        return false;
    }
}

function getServerLogFile(): string {
    return path.join(extensionContext.logUri.fsPath, 'server.log');
}

function getServerDebugEnvironment(): Record<string, string> {
    const config = vscode.workspace.getConfiguration("aiconls");
    const enabled = config.get<boolean>("server.debug.enabled", false);
    if (!enabled) {
        return {};
    }

    const port = config.get<number>("server.debug.port", 5678);
    const waitForClient = config.get<boolean>("server.debug.waitForClient", true);

    logger.info(
        `server debug enabled (debugpy attach on 127.0.0.1:${port}, waitForClient=${waitForClient})`
    );
    return {
        AICONLS_DEBUG: "1",
        AICONLS_DEBUG_HOST: "127.0.0.1",
        AICONLS_DEBUG_PORT: String(port),
        AICONLS_DEBUG_WAIT_FOR_CLIENT: waitForClient ? "1" : "0",
    };
}

async function getServerPath(pythonCommand: string[]): Promise<string | undefined> {
    const configuredServerPath = vscode.workspace
        .getConfiguration("aiconls")
        .get<string>("server.path");
    if (configuredServerPath) {
        if (await pathExists(configuredServerPath)) {
            return configuredServerPath;
        }

        const message = `Configured server path '${configuredServerPath}' does not exist.`;
        logger.error(message);
        void vscode.window.showErrorMessage(message);
        return;
    }

    const bundledServerPath = await getBundledServerPath();
    if (bundledServerPath) {
        return bundledServerPath;
    }

    const installedServerPath = path.join(getInstalledServerRoot(), SERVER_ENTRYPOINT);
    if (await pathExists(installedServerPath)) {
        return installedServerPath;
    }

    const installAction = "Install from Git";
    const response = await vscode.window.showErrorMessage(
        "AICON language server is not installed. Install it now from Git?",
        installAction
    );
    if (response !== installAction) {
        logger.error("Language server installation was cancelled by the user.");
        return;
    }

    await installOrUpdateServerFromGit(pythonCommand, getInstalledServerRoot());
    if (!(await pathExists(installedServerPath))) {
        throw new Error(
            `Language server entrypoint not found after installation at '${installedServerPath}'.`
        );
    }

    return installedServerPath;
}

async function getBundledServerPath(): Promise<string | undefined> {
    for (const folder of vscode.workspace.workspaceFolders ?? []) {
        const bundledServerPath = path.join(folder.uri.fsPath, BUNDLED_SERVER_PATH);
        if (await pathExists(bundledServerPath)) {
            return bundledServerPath;
        }
    }
    return;
}

/**
 * Whether the server's dependencies are in the selected environment. Those of the installed server are reinstalled
 * if missing; those of a bundled server belong to the repository's environment, so the user is told how to add them.
 */
async function ensureServerDependencies(
    pythonCommand: string[],
    serverPath: string,
    cwd: string
): Promise<boolean> {
    const installedServerPath = path.join(getInstalledServerRoot(), SERVER_ENTRYPOINT);
    const bundled = serverPath === (await getBundledServerPath());
    if (serverPath !== installedServerPath && !bundled) {
        return true;
    }

    const missingModules = await findMissingPythonModules(
        pythonCommand,
        REQUIRED_SERVER_MODULES,
        cwd
    );
    if (missingModules.length === 0) {
        return true;
    }

    if (bundled) {
        const message = [
            `The selected Python environment ('${pythonCommand[0]}') is missing the dependencies of the`,
            `language server in '${path.dirname(serverPath)}' (${missingModules.join(", ")}).`,
            "Install them into that environment (in the AICON repository: the `lint` extra, see tools/README.md)",
            "and restart the language server.",
        ].join(" ");
        logger.error(message);
        void vscode.window.showErrorMessage(message);
        return false;
    }

    logger.warn(
        [
            "Selected environment is missing server dependencies",
            `(${missingModules.join(", ")}).`,
            `Reinstalling into '${pythonCommand[0]}'.`,
        ].join(" ")
    );
    await installOrUpdateServerFromGit(pythonCommand, getInstalledServerRoot());

    const stillMissingModules = await findMissingPythonModules(
        pythonCommand,
        REQUIRED_SERVER_MODULES,
        cwd
    );
    if (stillMissingModules.length > 0) {
        throw new Error(
            [
                "Language server dependencies are still missing after reinstall:",
                `${stillMissingModules.join(", ")}.`,
                `Please install them in '${pythonCommand[0]}' and restart the language server.`,
            ].join(" ")
        );
    }
    return true;
}

function getPythonEnvironmentResource(): vscode.Uri | undefined {
    const activeDocument = vscode.window.activeTextEditor?.document;
    if (activeDocument?.languageId === "python" && activeDocument.uri.scheme === "file") {
        return activeDocument.uri;
    }

    return vscode.workspace.workspaceFolders?.[0]?.uri;
}

/**
 * Return the python command to use when starting the server.
 *
 * If debugging is enabled, this will also included the arguments to required
 * to wrap the server in a debug adapter.
 *
 * @returns The full python command needed in order to start the server.
 */
async function getPythonCommand(resource?: vscode.Uri): Promise<string[] | undefined> {
    const pythonCommand = await getPythonInterpreterCmd(resource);
    if (!pythonCommand) {
        return;
    }
    return pythonCommand;
}

/**
 * Return the command to use to start the python interpreter
 *
 * If no command is configured, this uses the official python extension to
 * grab the user's currently configured environment.
 *
 * @returns The command needed to invoke the python interpreter
 */
async function getPythonInterpreterCmd(resource?: vscode.Uri): Promise<string[] | undefined> {
    if (!python) {
        const message = "Python extension API is unavailable. Install/enable ms-python.python and reload VS Code.";
        logger.error(message);
        void vscode.window.showErrorMessage(message);
        return;
    }

    if (resource) {
        logger.info(`Looking for environment in which to execute: '${resource.toString()}'`);
    }
    // Use whichever python interpreter the user has configured.
    const activeEnvPath = python.environments.getActiveEnvironmentPath(resource);
    logger.info(`Found environment: ${activeEnvPath.id}: ${activeEnvPath.path}`);

    const activeEnv = await python.environments.resolveEnvironment(activeEnvPath);
    if (!activeEnv) {
        logger.error(`Unable to resolve envrionment: ${activeEnvPath}`);
        return;
    }

    const v = activeEnv.version;
    if(!v) {
        logger.error("No python version info");
        return;
    }
    const pythonVersion = semver.parse(`${v.major}.${v.minor}.${v.micro}`);

    if(!pythonVersion || !MIN_PYTHON) {
        logger.error("No python version info");
        return;
    }

    // Check to see if the environment satisfies the min Python version.
    if (semver.lt(pythonVersion, MIN_PYTHON)) {
        const message = [
            `Your currently configured environment provides Python v${pythonVersion} `,
            `but pygls requires v${MIN_PYTHON}.\n\nPlease choose another environment.`
        ].join('');

        const response = await vscode.window.showErrorMessage(message, "Change Environment");
        if (!response) {
            return;
        } else {
            await vscode.commands.executeCommand('python.setInterpreter');
            return;
        }
    }

    const pythonUri = activeEnv.executable.uri;
    if (!pythonUri) {
        logger.error(`URI of Python executable is undefined!`);
        return;
    }

    return [pythonUri.fsPath];
}

function getInstalledServerRoot(): string {
    return path.join(extensionContext.globalStorageUri.fsPath, INSTALLED_SERVER_DIR);
}

async function installOrUpdateServerFromGit(
    pythonCommand: string[],
    installRoot: string
): Promise<void> {
    const repository = vscode.workspace
        .getConfiguration("aiconls")
        .get<string>("server.gitRepository", DEFAULT_SERVER_GIT_REPOSITORY);
    const parentDir = path.dirname(installRoot);

    await vscode.window.withProgress(
        {
            location: vscode.ProgressLocation.Notification,
            title: "Installing AICON language server",
            cancellable: false,
        },
        async () => {
            await mkdir(parentDir, { recursive: true });

            const gitDir = path.join(installRoot, ".git");
            if (await pathExists(gitDir)) {
                await runCommand(
                    "git",
                    ["-C", installRoot, "pull", "--ff-only"],
                    parentDir
                );
            } else {
                if (await pathExists(installRoot)) {
                    throw new Error(
                        `Cannot install server because '${installRoot}' exists but is not a git repository.`
                    );
                }
                await runCommand(
                    "git",
                    ["clone", "--depth", "1", repository, installRoot],
                    parentDir
                );
            }

            await installServerPackage(pythonCommand, installRoot, parentDir);
        }
    );
}

/**
 * Install the server package and its dependencies into the selected environment.
 *
 * AICON environments are created with uv (`.venv`) or pixi (`.pixi/envs/<env>`),
 * neither of which ships pip, so uv is preferred and pip is only a fallback.
 */
async function installServerPackage(
    pythonCommand: string[],
    installRoot: string,
    cwd: string
): Promise<void> {
    const uvCommand = await findUvCommand(cwd);
    if (uvCommand) {
        await runCommand(
            uvCommand,
            [
                "pip", "install",
                "--python", pythonCommand[0],
                "--upgrade",
                "--reinstall-package", SERVER_PACKAGE,
                installRoot,
            ],
            cwd
        );
        return;
    }

    if (await commandSucceeds(pythonCommand[0], [...pythonCommand.slice(1), "-m", "pip", "--version"], cwd)) {
        await runCommand(
            pythonCommand[0],
            [...pythonCommand.slice(1), "-m", "pip", "install", "--upgrade", installRoot],
            cwd
        );
        return;
    }

    throw new Error(
        [
            `Cannot install the language server into '${pythonCommand[0]}':`,
            "uv was not found and the environment has no pip.",
            "Install uv (https://docs.astral.sh/uv/) and run 'AICONLS: Install/Update AICON Language Server' again.",
        ].join(" ")
    );
}

async function findUvCommand(cwd: string): Promise<string | undefined> {
    const candidates = [
        "uv",
        // Default locations of the standalone uv installer, which may be missing from VS Code's PATH.
        path.join(os.homedir(), ".local", "bin", "uv"),
        path.join(os.homedir(), ".cargo", "bin", "uv"),
    ];
    for (const candidate of candidates) {
        if (await commandSucceeds(candidate, ["--version"], cwd)) {
            return candidate;
        }
    }
    logger.info("uv not found, falling back to pip");
    return;
}

async function commandSucceeds(command: string, args: string[], cwd: string): Promise<boolean> {
    return await new Promise<boolean>((resolve) => {
        const proc = spawn(command, args, { cwd });
        proc.on("error", () => resolve(false));
        proc.on("close", (code) => resolve(code === 0));
    });
}

async function runCommand(command: string, args: string[], cwd: string): Promise<void> {
    await new Promise<void>((resolve, reject) => {
        const proc = spawn(command, args, { cwd });
        let stderr = "";

        proc.stdout.on("data", (data: Buffer) => {
            const text = data.toString();
            logger.info(text.trimEnd());
        });

        proc.stderr.on("data", (data: Buffer) => {
            const text = data.toString();
            stderr += text;
            logger.error(text.trimEnd());
        });

        proc.on("error", (error) => {
            reject(new Error(`Failed to run '${command} ${args.join(" ")}': ${error.message}`));
        });

        proc.on("close", (code) => {
            if (code === 0) {
                resolve();
                return;
            }
            reject(
                new Error(
                    `Command '${command} ${args.join(" ")}' failed with exit code ${code}. ${stderr.trim()}`
                )
            );
        });
    });
}

async function canImportPythonModule(
    pythonCommand: string[],
    moduleName: string,
    cwd: string
): Promise<boolean> {
    return await new Promise<boolean>((resolve, reject) => {
        const proc = spawn(
            pythonCommand[0],
            [...pythonCommand.slice(1), "-c", `import ${moduleName}`],
            { cwd }
        );
        let stderr = "";

        proc.stderr.on("data", (data: Buffer) => {
            stderr += data.toString();
        });

        proc.on("error", (error) => {
            reject(
                new Error(
                    `Failed to validate Python dependency '${moduleName}': ${error.message}`
                )
            );
        });

        proc.on("close", (code) => {
            if (code === 0) {
                resolve(true);
                return;
            }

            logger.info(
                `Dependency check failed for module '${moduleName}' in '${pythonCommand[0]}': ${stderr.trim()}`
            );
            resolve(false);
        });
    });
}

async function findMissingPythonModules(
    pythonCommand: string[],
    modules: readonly string[],
    cwd: string
): Promise<string[]> {
    const missingModules: string[] = [];
    for (const moduleName of modules) {
        const hasModule = await canImportPythonModule(pythonCommand, moduleName, cwd);
        if (!hasModule) {
            missingModules.push(moduleName);
        }
    }

    return missingModules;
}

async function pathExists(pathToCheck: string): Promise<boolean> {
    try {
        await access(pathToCheck, constants.F_OK);
        return true;
    } catch {
        return false;
    }
}

function getCwd() {
        const cwd = vscode.workspace.workspaceFolders?.[0].uri.fsPath;
	if(!cwd) {
		logger.error("No workspace open");
        throw new Error("No workspace opened!");
	}

    return cwd;
}
async function getPythonExtension(): Promise<PythonExtension | undefined> {
    try {
        return await PythonExtension.api();
    } catch (err) {
        logger.error(`Unable to load python extension: ${err}`);
        return;
    }
}
