import { constants } from "node:fs";
import { access, readFile, writeFile } from "node:fs/promises";
import * as path from "node:path";

import * as vscode from "vscode";
import { parseDocument, stringify } from "yaml";
import specVisualizerHtmlTemplate from "./specVisualizer.html";

type NodeKind = "component" | "connection" | "goal";
type EdgeKind = "connection-component" | "component-goal";

interface VisualizerNode {
    id: string;
    name: string;
    kind: NodeKind;
    label: string;
    position: {
        x: number;
        y: number;
    };
    details: Record<string, unknown>;
}

interface VisualizerEdge {
    id: string;
    source: string;
    target: string;
    kind: EdgeKind;
}

interface VisualizerData {
    specPath: string;
    diagnostics: string[];
    nodes: VisualizerNode[];
    edges: VisualizerEdge[];
    layoutConfig: LayoutConfig;
}

type LayoutAlgorithm = "layered" | "stress" | "force" | "radial" | "mrtree";
type LayoutDirection = "RIGHT" | "LEFT" | "DOWN" | "UP";
type LayoutEdgeRouting = "ORTHOGONAL" | "POLYLINE" | "SPLINES";

interface LayoutConfig {
    algorithm: LayoutAlgorithm;
    direction: LayoutDirection;
    edgeRouting: LayoutEdgeRouting;
    nodeSpacing: number;
    edgeNodeSpacing: number;
    layerSpacing: number;
    stressIterationLimit: number;
    forceIterations: number;
    forceRepulsivePower: number;
    radialCompaction: boolean;
}

interface RawSpecGoal {
    name?: unknown;
}

interface RawSpecComponent {
    component_type?: unknown;
    quantities?: unknown;
    goals?: unknown;
}

interface RawSpecConnection {
    components?: unknown;
}

interface RawSpecRoot {
    components?: unknown;
    goals?: unknown;
    connections?: unknown;
}

interface RefreshMessage {
    type: "refresh";
}

interface SaveNodeMessage {
    type: "saveNode";
    nodeId?: unknown;
    properties?: unknown;
}

interface CreateComponentMessage {
    type: "createComponent";
    name?: unknown;
}

interface CreateConnectionMessage {
    type: "createConnection";
    name?: unknown;
}

interface DeleteNodeMessage {
    type: "deleteNode";
    nodeId?: unknown;
}

interface RecalculateLayoutMessage {
    type: "recalculateLayout";
    layoutConfig?: unknown;
}

type VisualizerMessage =
    | RefreshMessage
    | SaveNodeMessage
    | CreateComponentMessage
    | CreateConnectionMessage
    | DeleteNodeMessage
    | RecalculateLayoutMessage;

let currentPanel: vscode.WebviewPanel | undefined;
let currentSpecPath: string | undefined;
let saveSubscription: vscode.Disposable | undefined;
const ElkConstructor = require("elkjs/lib/elk.bundled.js") as {
    new (): { layout: (graph: unknown) => Promise<{ children?: unknown[] }> };
};
const elk = new ElkConstructor();

const LAYOUT_NODE_WIDTH = 170;
const LAYOUT_NODE_HEIGHT = 54;
const DEFAULT_LAYOUT_CONFIG: LayoutConfig = {
    algorithm: "force",
    direction: "RIGHT",
    edgeRouting: "POLYLINE",
    nodeSpacing: 110,
    edgeNodeSpacing: 80,
    layerSpacing: 140,
    stressIterationLimit: 3000,
    forceIterations: 8000,
    forceRepulsivePower: 1.3,
    radialCompaction: true,
};
const FALLBACK_COLUMN_BY_KIND: Record<NodeKind, number> = {
    connection: 0,
    component: 1,
    goal: 2,
};

export function registerSpecVisualizerCommand(
    context: vscode.ExtensionContext,
    logger: vscode.LogOutputChannel
): void {
    context.subscriptions.push(
        vscode.commands.registerCommand("aiconls.visualizeSpec", async () => {
            const specPath = await resolveSpecPath();
            if (!specPath) {
                return;
            }
            let currentLayoutConfig: LayoutConfig = { ...DEFAULT_LAYOUT_CONFIG };

            const data = await buildVisualizerData(specPath, currentLayoutConfig);
            if (currentPanel && currentSpecPath === specPath) {
                currentPanel.reveal(vscode.ViewColumn.Beside, true);
                await currentPanel.webview.postMessage({ type: "update", data });
                return;
            }

            currentPanel?.dispose();

            currentSpecPath = specPath;
            currentPanel = vscode.window.createWebviewPanel(
                "aiconlsSpecVisualizer",
                `AICON Spec Visualizer: ${path.basename(specPath)}`,
                vscode.ViewColumn.Beside,
                {
                    enableScripts: true,
                    retainContextWhenHidden: true,
                }
            );
            currentPanel.webview.html = getWebviewHtml(data);

            const refreshFromDisk = async (): Promise<void> => {
                if (!currentPanel || !currentSpecPath) {
                    return;
                }
                const refreshedData = await buildVisualizerData(
                    currentSpecPath,
                    currentLayoutConfig
                );
                await currentPanel.webview.postMessage({ type: "update", data: refreshedData });
            };

            currentPanel.webview.onDidReceiveMessage(
                async (message: VisualizerMessage) => {
                    if (!currentSpecPath || !currentPanel) {
                        return;
                    }
                    try {
                        if (message.type === "recalculateLayout") {
                            currentLayoutConfig = parseLayoutConfig(
                                message.layoutConfig,
                                currentLayoutConfig
                            );
                            const relayoutData = await buildVisualizerData(
                                currentSpecPath,
                                currentLayoutConfig
                            );
                            await currentPanel.webview.postMessage({ type: "update", data: relayoutData });
                            await currentPanel.webview.postMessage({
                                type: "status",
                                kind: "info",
                                message: "Layout recalculated.",
                            });
                            return;
                        }
                        await handleWebviewMessage(message, currentSpecPath);
                        if (message.type !== "refresh") {
                            await currentPanel.webview.postMessage({
                                type: "status",
                                kind: "info",
                                message: "spec.yaml updated.",
                            });
                        }
                        await refreshFromDisk();
                    } catch (error) {
                        const errorMessage =
                            error instanceof Error ? error.message : String(error);
                        logger.error(`spec visualizer action failed: ${errorMessage}`);
                        await currentPanel.webview.postMessage({
                            type: "status",
                            kind: "error",
                            message: errorMessage,
                        });
                    }
                },
                undefined,
                context.subscriptions
            );

            currentPanel.onDidDispose(
                () => {
                    currentPanel = undefined;
                    currentSpecPath = undefined;
                    saveSubscription?.dispose();
                    saveSubscription = undefined;
                },
                undefined,
                context.subscriptions
            );

            saveSubscription?.dispose();
            saveSubscription = vscode.workspace.onDidSaveTextDocument(
                async (document) => {
                    if (!currentSpecPath || document.uri.fsPath !== currentSpecPath) {
                        return;
                    }
                    try {
                        await refreshFromDisk();
                    } catch (error) {
                        logger.error(`Failed to refresh visualizer after save: ${String(error)}`);
                    }
                }
            );
            context.subscriptions.push(saveSubscription);
        })
    );
}

async function resolveSpecPath(): Promise<string | undefined> {
    const activePath = vscode.window.activeTextEditor?.document.uri.fsPath;
    if (activePath && path.basename(activePath).toLowerCase() === "spec.yaml") {
        return activePath;
    }

    const workspaceRoot = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
    if (!workspaceRoot) {
        void vscode.window.showErrorMessage("No workspace is open.");
        return;
    }

    const configuredProjectRoot = vscode.workspace
        .getConfiguration("aiconls")
        .get<string>("project.root");
    if (configuredProjectRoot) {
        const resolvedProjectRoot = configuredProjectRoot.replace("${workspace}", workspaceRoot);
        const configuredSpecPath = path.resolve(resolvedProjectRoot, "spec.yaml");
        if (await pathExists(configuredSpecPath)) {
            return configuredSpecPath;
        }
    }

    const workspaceSpecPath = path.join(workspaceRoot, "spec.yaml");
    if (await pathExists(workspaceSpecPath)) {
        return workspaceSpecPath;
    }

    void vscode.window.showErrorMessage(
        "Could not find spec.yaml. Open spec.yaml or place it in the configured AICON project root."
    );
    return;
}

async function pathExists(pathToCheck: string): Promise<boolean> {
    try {
        await access(pathToCheck, constants.F_OK);
        return true;
    } catch {
        return false;
    }
}

function asRecord(value: unknown): Record<string, unknown> | undefined {
    if (!value || typeof value !== "object" || Array.isArray(value)) {
        return;
    }
    return value as Record<string, unknown>;
}

function asStringArray(value: unknown): string[] {
    if (!Array.isArray(value)) {
        return [];
    }
    return value.filter((entry): entry is string => typeof entry === "string");
}

function requireRecord(value: unknown, context: string): Record<string, unknown> {
    const mapped = asRecord(value);
    if (!mapped) {
        throw new Error(context);
    }
    return mapped;
}

function parseNodeId(nodeId: string): { kind: NodeKind; name: string } {
    if (nodeId.startsWith("component:")) {
        return { kind: "component", name: nodeId.slice("component:".length) };
    }
    if (nodeId.startsWith("connection:")) {
        return { kind: "connection", name: nodeId.slice("connection:".length) };
    }
    if (nodeId.startsWith("goal:")) {
        return { kind: "goal", name: nodeId.slice("goal:".length) };
    }
    throw new Error(`Invalid node id '${nodeId}'.`);
}

function normalizeName(name: unknown, context: string): string {
    if (typeof name !== "string") {
        throw new Error(`${context} must be a string.`);
    }
    const trimmed = name.trim();
    if (!trimmed) {
        throw new Error(`${context} cannot be empty.`);
    }
    return trimmed;
}

function clampNumber(value: number, min: number, max: number): number {
    return Math.max(min, Math.min(max, value));
}

function parseNumber(value: unknown, fallback: number): number {
    if (typeof value !== "number" || Number.isNaN(value) || !Number.isFinite(value)) {
        return fallback;
    }
    return value;
}

function parseLayoutConfig(value: unknown, fallback: LayoutConfig): LayoutConfig {
    const raw = asRecord(value);
    const algorithmSet = new Set<LayoutAlgorithm>([
        "layered",
        "stress",
        "force",
        "radial",
        "mrtree",
    ]);
    const directionSet = new Set<LayoutDirection>(["RIGHT", "LEFT", "DOWN", "UP"]);
    const routingSet = new Set<LayoutEdgeRouting>(["ORTHOGONAL", "POLYLINE", "SPLINES"]);

    const algorithm =
        typeof raw?.algorithm === "string" && algorithmSet.has(raw.algorithm as LayoutAlgorithm)
            ? (raw.algorithm as LayoutAlgorithm)
            : fallback.algorithm;
    const direction =
        typeof raw?.direction === "string" && directionSet.has(raw.direction as LayoutDirection)
            ? (raw.direction as LayoutDirection)
            : fallback.direction;
    const edgeRouting =
        typeof raw?.edgeRouting === "string" &&
        routingSet.has(raw.edgeRouting as LayoutEdgeRouting)
            ? (raw.edgeRouting as LayoutEdgeRouting)
            : fallback.edgeRouting;
    const radialCompaction =
        typeof raw?.radialCompaction === "boolean"
            ? raw.radialCompaction
            : fallback.radialCompaction;

    return {
        algorithm,
        direction,
        edgeRouting,
        nodeSpacing: clampNumber(parseNumber(raw?.nodeSpacing, fallback.nodeSpacing), 20, 500),
        edgeNodeSpacing: clampNumber(
            parseNumber(raw?.edgeNodeSpacing, fallback.edgeNodeSpacing),
            10,
            300
        ),
        layerSpacing: clampNumber(parseNumber(raw?.layerSpacing, fallback.layerSpacing), 20, 500),
        stressIterationLimit: clampNumber(
            Math.round(parseNumber(raw?.stressIterationLimit, fallback.stressIterationLimit)),
            200,
            8000
        ),
        forceIterations: clampNumber(
            Math.round(parseNumber(raw?.forceIterations, fallback.forceIterations)),
            100,
            8000
        ),
        forceRepulsivePower: clampNumber(
            parseNumber(raw?.forceRepulsivePower, fallback.forceRepulsivePower),
            0.1,
            10
        ),
        radialCompaction,
    };
}

function normalizeShape(shape: unknown): string {
    if (typeof shape === "string" && shape.trim().length > 0) {
        return shape.trim();
    }
    if (
        Array.isArray(shape) &&
        shape.every(
            (entry) =>
                typeof entry === "number" &&
                Number.isFinite(entry) &&
                Number.isInteger(entry)
        )
    ) {
        return `(${shape.join(", ")})`;
    }
    throw new Error(
        "Quantity shapes must be either tuple-like strings (e.g. '(1, 4)') or integer arrays."
    );
}

function normalizeComponentProperties(value: unknown): RawSpecComponent {
    const properties = requireRecord(
        value,
        "Component properties must be an object with component_type, goals, and quantities."
    );
    const componentType = normalizeName(properties.component_type, "component_type");
    const goals = asStringArray(properties.goals).map((entry) => entry.trim()).filter(Boolean);

    const quantitiesValue = requireRecord(
        properties.quantities,
        "Component quantities must be an object."
    );
    const normalizedQuantities: Record<string, { shape: string }> = {};
    for (const [quantityName, quantityValue] of Object.entries(quantitiesValue)) {
        const normalizedQuantityName = normalizeName(quantityName, "Quantity name");
        const quantityMap = asRecord(quantityValue);
        const shape = normalizeShape(quantityMap ? quantityMap.shape : quantityValue);
        normalizedQuantities[normalizedQuantityName] = { shape };
    }

    return {
        component_type: componentType,
        goals,
        quantities: normalizedQuantities,
    };
}

function normalizeConnectionProperties(value: unknown): RawSpecConnection {
    const properties = requireRecord(
        value,
        "Connection properties must be an object containing a components array."
    );
    const components = asStringArray(properties.components)
        .map((entry) => entry.trim())
        .filter(Boolean);
    return { components };
}

function normalizeGoalProperties(value: unknown): RawSpecGoal {
    const properties = requireRecord(value, "Goal properties must be an object.");
    const name = normalizeName(properties.name, "Goal name");
    return { name };
}

function ensureSection(root: RawSpecRoot, section: keyof RawSpecRoot): Record<string, unknown> {
    const current = asRecord(root[section]);
    if (current) {
        return current;
    }
    const next: Record<string, unknown> = {};
    root[section] = next;
    return next;
}

async function readSpecRootForEditing(specPath: string): Promise<RawSpecRoot> {
    const source = await readFile(specPath, "utf-8");
    const parsed = parseDocument(source);
    if (parsed.errors.length > 0) {
        const details = parsed.errors.map((error) => error.message).join(" | ");
        throw new Error(`Cannot edit malformed spec.yaml: ${details}`);
    }
    const parsedJson = parsed.toJSON() as unknown;
    if (parsedJson === null || parsedJson === undefined) {
        return {};
    }
    return requireRecord(parsedJson, "spec.yaml root must be a YAML mapping.");
}

async function saveSpecRoot(specPath: string, root: RawSpecRoot): Promise<void> {
    const serialized = stringify(root);
    const finalText = serialized.endsWith("\n") ? serialized : `${serialized}\n`;
    await writeFile(specPath, finalText, "utf-8");
}

async function upsertNodeProperties(
    specPath: string,
    nodeId: string,
    properties: unknown
): Promise<void> {
    const root = await readSpecRootForEditing(specPath);
    const parsedNode = parseNodeId(nodeId);

    if (parsedNode.kind === "component") {
        const components = ensureSection(root, "components");
        if (!(parsedNode.name in components)) {
            throw new Error(`Component '${parsedNode.name}' does not exist.`);
        }
        components[parsedNode.name] = normalizeComponentProperties(properties);
        await saveSpecRoot(specPath, root);
        return;
    }

    if (parsedNode.kind === "connection") {
        const connections = ensureSection(root, "connections");
        if (!(parsedNode.name in connections)) {
            throw new Error(`Connection '${parsedNode.name}' does not exist.`);
        }
        connections[parsedNode.name] = normalizeConnectionProperties(properties);
        await saveSpecRoot(specPath, root);
        return;
    }

    const goals = ensureSection(root, "goals");
    if (!(parsedNode.name in goals)) {
        throw new Error(`Goal '${parsedNode.name}' does not exist.`);
    }
    goals[parsedNode.name] = normalizeGoalProperties(properties);
    await saveSpecRoot(specPath, root);
}

async function createComponent(specPath: string, name: string): Promise<void> {
    const root = await readSpecRootForEditing(specPath);
    const components = ensureSection(root, "components");
    if (name in components) {
        throw new Error(`Component '${name}' already exists.`);
    }
    components[name] = {
        component_type: "Sensor",
        goals: [],
        quantities: {},
    } as RawSpecComponent;
    await saveSpecRoot(specPath, root);
}

async function createConnection(specPath: string, name: string): Promise<void> {
    const root = await readSpecRootForEditing(specPath);
    const connections = ensureSection(root, "connections");
    if (name in connections) {
        throw new Error(`Connection '${name}' already exists.`);
    }
    connections[name] = {
        components: [],
    } as RawSpecConnection;
    await saveSpecRoot(specPath, root);
}

async function deleteNode(specPath: string, nodeId: string): Promise<void> {
    const root = await readSpecRootForEditing(specPath);
    const parsedNode = parseNodeId(nodeId);

    if (parsedNode.kind === "component") {
        const components = ensureSection(root, "components");
        if (!(parsedNode.name in components)) {
            throw new Error(`Component '${parsedNode.name}' does not exist.`);
        }
        delete components[parsedNode.name];

        const connections = ensureSection(root, "connections");
        for (const connectionValue of Object.values(connections)) {
            const connection = asRecord(connectionValue);
            if (!connection) {
                continue;
            }
            connection.components = asStringArray(connection.components).filter(
                (componentName) => componentName !== parsedNode.name
            );
        }
        await saveSpecRoot(specPath, root);
        return;
    }

    if (parsedNode.kind === "connection") {
        const connections = ensureSection(root, "connections");
        if (!(parsedNode.name in connections)) {
            throw new Error(`Connection '${parsedNode.name}' does not exist.`);
        }
        delete connections[parsedNode.name];
        await saveSpecRoot(specPath, root);
        return;
    }

    throw new Error("Goal deletion is not supported from this view.");
}

async function handleWebviewMessage(message: VisualizerMessage, specPath: string): Promise<void> {
    if (message.type === "refresh") {
        return;
    }

    if (message.type === "saveNode") {
        const nodeId = normalizeName(message.nodeId, "Node id");
        await upsertNodeProperties(specPath, nodeId, message.properties);
        return;
    }

    if (message.type === "createComponent") {
        const name = normalizeName(message.name, "Component name");
        await createComponent(specPath, name);
        return;
    }

    if (message.type === "createConnection") {
        const name = normalizeName(message.name, "Connection name");
        await createConnection(specPath, name);
        return;
    }

    if (message.type === "deleteNode") {
        const nodeId = normalizeName(message.nodeId, "Node id");
        await deleteNode(specPath, nodeId);
        return;
    }

    throw new Error("Unknown webview message.");
}

function applyFallbackLayout(nodes: VisualizerNode[]): VisualizerNode[] {
    const byKind: Record<NodeKind, VisualizerNode[]> = {
        component: [],
        connection: [],
        goal: [],
    };
    for (const node of nodes) {
        byKind[node.kind].push(node);
    }
    for (const nodeGroup of Object.values(byKind)) {
        nodeGroup.sort((left, right) => left.label.localeCompare(right.label));
    }

    const verticalGap = 90;
    const horizontalGap = 260;
    const yPadding = 70;
    const xPadding = 80;
    const positionById = new Map<string, { x: number; y: number }>();
    for (const kind of ["connection", "component", "goal"] as const) {
        const column = FALLBACK_COLUMN_BY_KIND[kind];
        const x = xPadding + column * horizontalGap;
        byKind[kind].forEach((node, index) => {
            positionById.set(node.id, {
                x,
                y: yPadding + index * verticalGap,
            });
        });
    }

    return nodes.map((node) => ({
        ...node,
        position: positionById.get(node.id) ?? { x: 0, y: 0 },
    }));
}

async function applyElkLayout(
    nodes: VisualizerNode[],
    edges: VisualizerEdge[],
    layoutConfig: LayoutConfig,
    diagnostics: string[]
): Promise<VisualizerNode[]> {
    if (nodes.length === 0) {
        return nodes;
    }

    const elkNodes = nodes.map((node) => ({
        id: node.id,
        width: LAYOUT_NODE_WIDTH,
        height: LAYOUT_NODE_HEIGHT,
    }));
    const elkEdges = edges.map((edge) => ({
        id: edge.id,
        sources: [edge.source],
        targets: [edge.target],
    }));

    try {
        const layoutOptions: Record<string, string> = {
            "elk.algorithm": layoutConfig.algorithm,
            "elk.padding": "[top=36,left=36,bottom=36,right=36]",
            "elk.spacing.nodeNode": String(layoutConfig.nodeSpacing),
            "elk.spacing.edgeNode": String(layoutConfig.edgeNodeSpacing),
            "elk.edgeRouting": layoutConfig.edgeRouting,
        };

        if (layoutConfig.algorithm === "layered") {
            layoutOptions["elk.direction"] = layoutConfig.direction;
            layoutOptions["elk.layered.spacing.nodeNodeBetweenLayers"] = String(
                layoutConfig.layerSpacing
            );
            layoutOptions["elk.layered.crossingMinimization.strategy"] = "LAYER_SWEEP";
            layoutOptions["elk.layered.nodePlacement.strategy"] = "NETWORK_SIMPLEX";
        }

        if (layoutConfig.algorithm === "stress") {
            layoutOptions["elk.stress.iterationLimit"] = String(
                layoutConfig.stressIterationLimit
            );
        }

        if (layoutConfig.algorithm === "force") {
            layoutOptions["elk.force.iterations"] = String(layoutConfig.forceIterations);
            layoutOptions["elk.force.repulsivePower"] = String(
                layoutConfig.forceRepulsivePower
            );
        }

        if (layoutConfig.algorithm === "radial") {
            layoutOptions["elk.radial.compactionStepSize"] = layoutConfig.radialCompaction
                ? "0.8"
                : "0";
        }

        const layout = await elk.layout({
            id: "aicon-spec-graph",
            layoutOptions,
            children: elkNodes,
            edges: elkEdges,
        });

        const positions = new Map<string, { x: number; y: number }>();
        for (const child of layout.children ?? []) {
            const childMap = asRecord(child);
            if (!childMap || typeof childMap.id !== "string") {
                continue;
            }
            const x = typeof childMap.x === "number" ? childMap.x : 0;
            const y = typeof childMap.y === "number" ? childMap.y : 0;
            const width =
                typeof childMap.width === "number" ? childMap.width : LAYOUT_NODE_WIDTH;
            const height =
                typeof childMap.height === "number" ? childMap.height : LAYOUT_NODE_HEIGHT;
            positions.set(childMap.id, {
                x: x + width / 2,
                y: y + height / 2,
            });
        }

        if (positions.size !== nodes.length) {
            diagnostics.push(
                "ELK layout returned incomplete node positions; using fallback layout."
            );
            return applyFallbackLayout(nodes);
        }

        return nodes.map((node) => ({
            ...node,
            position: positions.get(node.id) ?? { x: 0, y: 0 },
        }));
    } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        diagnostics.push(`ELK layout failed: ${message}`);
        return applyFallbackLayout(nodes);
    }
}

async function buildVisualizerData(
    specPath: string,
    layoutConfig: LayoutConfig
): Promise<VisualizerData> {
    const diagnostics: string[] = [];
    const source = await readFile(specPath, "utf-8");
    const parsed = parseDocument(source);
    if (parsed.errors.length > 0) {
        diagnostics.push(...parsed.errors.map((error) => error.message));
    }

    const root = asRecord(parsed.toJSON() as RawSpecRoot | undefined);
    if (!root) {
        return { specPath, diagnostics, nodes: [], edges: [], layoutConfig };
    }

    const rawComponents = asRecord(root.components);
    const rawGoals = asRecord(root.goals);
    const rawConnections = asRecord(root.connections);

    if (!rawComponents) {
        diagnostics.push("Top-level field 'components' should be a mapping.");
    }
    if (!rawGoals) {
        diagnostics.push("Top-level field 'goals' should be a mapping.");
    }
    if (!rawConnections) {
        diagnostics.push("Top-level field 'connections' should be a mapping.");
    }

    const nodes: VisualizerNode[] = [];
    const edges: VisualizerEdge[] = [];

    const componentNames = new Set<string>();
    const goalNames = new Set<string>();

    for (const [goalId, rawGoal] of Object.entries(rawGoals ?? {})) {
        goalNames.add(goalId);
        const goal = asRecord(rawGoal) as RawSpecGoal | undefined;
        const displayName =
            typeof goal?.name === "string" && goal.name.trim().length > 0 ? goal.name : goalId;

        nodes.push({
            id: `goal:${goalId}`,
            name: goalId,
            kind: "goal",
            label: displayName,
            position: { x: 0, y: 0 },
            details: {
                name: displayName,
            },
        });
    }

    for (const [componentName, rawComponent] of Object.entries(rawComponents ?? {})) {
        componentNames.add(componentName);
        const component = asRecord(rawComponent) as RawSpecComponent | undefined;
        if (!component) {
            diagnostics.push(`Component '${componentName}' should be a mapping.`);
            continue;
        }

        const componentType =
            typeof component.component_type === "string"
                ? component.component_type
                : "Unknown";

        const quantitiesRecord = asRecord(component.quantities);
        const quantityDetails: Record<string, unknown> = {};
        for (const [quantityName, quantityValue] of Object.entries(quantitiesRecord ?? {})) {
            const quantity = asRecord(quantityValue);
            quantityDetails[quantityName] = { shape: quantity?.shape ?? "(missing shape)" };
        }

        const goals = asStringArray(component.goals);

        nodes.push({
            id: `component:${componentName}`,
            name: componentName,
            kind: "component",
            label: componentName,
            position: { x: 0, y: 0 },
            details: {
                component_type: componentType,
                goals,
                quantities: quantityDetails,
            },
        });

        for (const goalName of goals) {
            if (!goalNames.has(goalName)) {
                diagnostics.push(
                    `Component '${componentName}' references unknown goal '${goalName}'.`
                );
                continue;
            }
            edges.push({
                id: `component-goal:${componentName}->${goalName}`,
                source: `component:${componentName}`,
                target: `goal:${goalName}`,
                kind: "component-goal",
            });
        }
    }

    for (const [connectionName, rawConnection] of Object.entries(rawConnections ?? {})) {
        const connection = asRecord(rawConnection) as RawSpecConnection | undefined;
        if (!connection) {
            diagnostics.push(`Connection '${connectionName}' should be a mapping.`);
            continue;
        }

        const connectedComponents = asStringArray(connection.components);
        nodes.push({
            id: `connection:${connectionName}`,
            name: connectionName,
            kind: "connection",
            label: connectionName,
            position: { x: 0, y: 0 },
            details: {
                components: connectedComponents,
            },
        });

        for (const componentName of connectedComponents) {
            if (!componentNames.has(componentName)) {
                diagnostics.push(
                    `Connection '${connectionName}' references unknown component '${componentName}'.`
                );
                continue;
            }
            edges.push({
                id: `connection-component:${connectionName}->${componentName}`,
                source: `connection:${connectionName}`,
                target: `component:${componentName}`,
                kind: "connection-component",
            });
        }
    }

    const positionedNodes = await applyElkLayout(nodes, edges, layoutConfig, diagnostics);
    return { specPath, diagnostics, nodes: positionedNodes, edges, layoutConfig };
}

function getWebviewHtml(initialData: VisualizerData): string {
    const nonce = getNonce();
    const serialized = JSON.stringify(initialData).replace(/</g, "\\u003c");
    return specVisualizerHtmlTemplate
        .replaceAll("__AICONLS_SPEC_VIS_NONCE__", nonce)
        .replace("__AICONLS_SPEC_VIS_INITIAL_DATA__", serialized);
}

function getNonce(): string {
    const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    let nonce = "";
    for (let i = 0; i < 32; i += 1) {
        nonce += alphabet.charAt(Math.floor(Math.random() * alphabet.length));
    }
    return nonce;
}
