export const Formats: import('../sim/dex-formats').FormatList = [
	{
		section: "Custom",
	},
	{
		// smogon-bench-v1: the randbats ruleset minus 'Obtainable' (which rejects some
		// randbats sets), with no `team` key so clients can bring pre-generated teams.
		name: "[Gen 4] Random Battle Fixed Teams",
		mod: 'gen4',
		ruleset: ['Sleep Clause Mod', 'HP Percentage Mod', 'Cancel Mod'],
	},
];
