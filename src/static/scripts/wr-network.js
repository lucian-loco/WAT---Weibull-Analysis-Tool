var options = {
    nodes: {
        borderWidth: 1,
        borderWidthSelected: 2,
        //brokenImage:undefined,
        //chosen: true,
        color: {
            border: '#094909',
            background: '#0d6d0d',
            highlight: {
            border: '#fb6d48',
            background: '#ffaf45'
        },
        hover: {
            border: '#25e425',
            background: '#19c819'
        }
        },
        font: {
            color: '#ffffff',
            face: 'sans-serif',
        },
        labelHighlightBold: false,
        shape: "box",
        //shadow: {
        //  enabled: true
        //}
    },

    edges: {
        color: {
            color: '#0d6d0d',
            hover: '#19c819',
            highlight: '#ffaf45',
        },
        smooth: {
            type: "cubicBezier",
            roundness: 0.5,
            forceDirection: "horizontal",
        },
    },

    groups: {
        stateError: { color: { background: 'red' } },
        stateWarning: { color: { background: 'orange' } },
        stateOk: { color: { background: 'green' } },
    },

    layout: {
        hierarchical: {
        direction: "LR",	// left-right
        levelSeparation: 250,	// distance between levels
        nodeSpacing: 100,	// min distance between nodes
        treeSpacing: 200,	// distance between independent trees
        //parentCentralization: false,
        //sortMethod: "directed",
        //shakeTowards: "roots",
        },
    },

    interaction: {
        dragNodes: false,
        hover: true,
        navigationButtons: true,
        multiselect: false,
    },

    physics: false,
};

var container = document.getElementById('network');
var nodes = new vis.DataSet();
const nodesView = new vis.DataView(nodes);
var edges = new vis.DataSet();
const edgesView = new vis.DataView(edges);
var network = new vis.Network(container, { nodes: nodesView, edges: edgesView }, options);

// Get the source parameter from the script URL
const currentScript = document.currentScript;
const src = currentScript.src;
const params = new URL(src).searchParams;
const connectivity_source = params.get('src') ? params.get('src') : 'ptp';

// Function to fetch data from another address
function updateData() {
fetch('/whiterabbit/connections/' + connectivity_source)
    .then(response => response.json())
    .then(data => {
    // Clear existing data
    nodes.clear();
    edges.clear();

    // Add new nodes and edges
    nodes.add(data.nodes);
    edges.add(data.edges);
    })
    .catch(error => console.error('Error fetching data:', error));
}

// Initial fetch and update
updateData();

// Fetch and update every 15 seconds
setInterval(updateData, 15000);