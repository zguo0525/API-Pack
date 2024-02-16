/*
INSTRUCTIONS: 
- Run the script with argumets for "input_file"
- Important: DO NOT INCLUDE THE EXTENSION!

EXAMPLES: 
    Node generate_api_calls.js ./ 'Aspera ATS API-1.0.3'
    Node step1_generate_api_calls.js /Users/amezasor/Projects/openapi-snippet/test/ 'api_gurus_1forge.com_0.0.1_swagger'
*/

function load_targets(config_file){
    var targets = fs.readFileSync(config_file).toString().split("\n");
    return targets;
}

// Load input data
function load_input(input_file){
    return require(input_file);
}

// Save output as json file
function save_file(output_file, output) {
    const dir = path.dirname(output_file);

    // Check if the directory exists, if not, create it
    if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
    }

    // Now that the directory is ensured to exist, write the file
    fs.writeFile(output_file, output, (error) => {
        if (error) {
            console.error(error);
            throw error; // Throw error to handle it outside if necessary
        }
        console.log(`${output_file} was saved!`);
    });
}

// Generate snippets for each endpoint
function generate_api_calls(data, targets, input_file){
    try {
        for(var path in data["paths"]){
            // console.log(path)
            for(var method in data["paths"][path]){
                results_snippets = OpenAPISnippet.getEndpointSnippets(data, path, method, targets);
                data["paths"][path][method]["api_calls"] = results_snippets["snippets"];
            }
        }
        // converting the JSON object to a string
        return JSON.stringify(data, undefined, 4);
    
    } catch (err) {
      console.log(`ERROR: cannot generate api calls for ${input_file} ${err}`)
    }
    return "";
}

// Req arguments
const OpenAPISnippet = require('openapi-snippet');
var fs = require('fs');
var path = require('path');
var arguments = process.argv;

// Directories
const MAIN_DIR = arguments[2]
const FOLDER = arguments[4] 
// const INPUT_DIR = `${MAIN_DIR}input_raw/`;
const INPUT_DIR = `${MAIN_DIR}${FOLDER}/`;

const CONFIG_FILES_DIR = `${MAIN_DIR}config_files/`;
const OUTPUT_DIR = `./data/input/${FOLDER}/generated/`;


// File Names
const INPUT_FILE =  arguments[3];
const OUTPUT_FILE = `${OUTPUT_DIR}${INPUT_FILE.replace(/\s/g, '')}_api_calls.json`;

// Call functions here
// targets = load_targets(`${CONFIG_FILES_DIR}${INPUT_FILE}.txt`);
targets = load_targets(`${CONFIG_FILES_DIR}10_base.txt`);
data = load_input(`${INPUT_DIR}${INPUT_FILE}.json`);
output = generate_api_calls(data, targets, INPUT_FILE);

// Summary
console.log(`Input directory path: ${INPUT_DIR}`);
console.log(`Configuration files directory path: ${CONFIG_FILES_DIR}`);
console.log(`Output directory path: ${OUTPUT_DIR}`);
console.log(`Input file name (do not inlcude extension): ${INPUT_FILE}.json`);
console.log(`Name of file with target languages (do not include extension): ${CONFIG_FILES_DIR}10_base.txt`);

if (typeof output === "string" && output.length === 0) {
    console.log(`${OUTPUT_FILE} was NOT saved!`);
}else{
    save_file(OUTPUT_FILE, output);
}
console.log("-----------------")