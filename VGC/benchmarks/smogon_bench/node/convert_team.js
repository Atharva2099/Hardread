// Reads a packed team on stdin, prints the export-format team on stdout.
// Usage: node convert_team.js <showdown_dir>
const path = require('path');
const dir = path.resolve(process.argv[2]);
const {Teams} = require(path.join(dir, 'dist', 'sim'));
let s = '';
process.stdin.on('data', d => s += d).on('end', () => {
	process.stdout.write(Teams.export(Teams.unpack(s.trim())));
});
