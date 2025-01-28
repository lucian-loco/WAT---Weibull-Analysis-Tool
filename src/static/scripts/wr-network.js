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
        levelSeparation: 400,	// distance between levels
        nodeSpacing: 75,	// min distance between nodes
        treeSpacing: 250,	// distance between independent trees
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
function updateData(first_update) {
    first_update = first_update || false;

    fetch('/whiterabbit/connections/' + connectivity_source)
        .then(response => response.json())
        .then(data => {
            if (!first_update) {
                nodes.clear();
                edges.clear();
            }

            nodes.add(data.nodes);
            edges.add(data.edges);

            if (first_update) {
                network.fit();
            }

            // TODO make it work
            // TODO nodes.updateOnly(data.nodes);
            // TODO edges.updateOnly(data.edges);
    })
    .catch(error => console.error('Error fetching data:', error));
}


// Assign a callback function for the searchbox
const input = document.getElementById('search');

input.addEventListener('input', function () {
    const searchValue = input.value;
    const animProps = { duration: 750, easingFunction: 'easeInQuad' }

    ids = nodes.getIds({
        filter: function (item) {
            return item.label.includes(searchValue);
        }
    });

    if (ids.length == 1) {
        // network.selectNodes(ids);
        network.focus(ids[0], { position: { x: 0, y: 0 }, scale: 1.5, animation: animProps });
    }
});


updateData(true);   // Initial fetch and update

// Fetch and update every 15 seconds
setInterval(updateData, 15000);