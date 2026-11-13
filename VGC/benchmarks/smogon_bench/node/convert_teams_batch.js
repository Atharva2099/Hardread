// Reads a JSON list of packed teams on stdin, prints a JSON list of export strings.
// Usage: node convert_teams_batch.js <showdown_dir>
const path = require('path');
const dir = path.resolve(process.argv[2]);
const {Teams} = require(path.join(dir, 'dist', 'sim'));
let s = '';
process.stdin.on('data', d => s += d).on('end', () => {
	const packed = JSON.parse(s);
	process.stdout.write(JSON.stringify(packed.map(p => Teams.export(Teams.unpack(p.trim())))));
});
