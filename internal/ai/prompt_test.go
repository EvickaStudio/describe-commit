package ai

import (
	"strings"
	"testing"
)

func TestGeneratePrompt(t *testing.T) {
	for name, tc := range map[string]struct {
		giveOpts     options
		wantContains []string
		wantNot      []string
	}{
		"short without emoji": {
			giveOpts: options{
				ShortMessageOnly: false,
				EnableEmoji:      false,
			},
			wantContains: []string{
				"Role", "acting as a Git",
				"Task", "git diff --staged", "convert it", "well-structured Git commit message",
				"Guidelines",
				"Conventional Commit",
				"`<type>(<scope>): <message>`", "fix(auth): Resolve",
				"Commit Message Structure",
				"Focus on summarizing", "Vague messages like", "Use present tense",
				"The first line should follow the Conventional Commit format",
				"Keep commit messages clean",
				"without period at the end",
				"Commit Body",
				"add a detailed description in bullet points",
				"Explain additional context",
				"Include a **summary** and a list",
				"Avoid excessive detail",
				"Don't start it with \"This commit\"",
				"Security", "Never include sensitive data",
			},
			wantNot: []string{
				"<emoji>", "♻️",
				"Summarize all changes in a single",
			},
		},
		"long with emoji": {
			giveOpts: options{
				ShortMessageOnly: true,
				EnableEmoji:      true,
			},
			wantContains: []string{
				"Role", "acting as a Git",
				"Task", "git diff --staged", "convert it", "well-structured Git commit message",
				"Guidelines",
				"Conventional Commit",
				"`<emoji> <type>(<scope>): <message>`", "🐛 fix(auth): Resolve",
				"Focus on the primary purpose of the commit",
				"Summarize all changes in a single",
				"Explain why the changes were made",
				"Security", "Never include sensitive data",
			},
			wantNot: []string{
				"`<type>(<scope>): <message>`",
				"Commit Message Structure",
				"Focus on summarizing",
				"Vague messages like",
				"Use present tense",
				"The first line should follow the Conventional Commit format",
				"Keep commit messages clean",
				"without period at the end",
				"Commit Body",
				"add a detailed description in bullet points",
				"Explain additional context",
				"Include a **summary** and a list",
				"Avoid excessive detail",
				"Don't start it with \"This commit\"",
			},
		},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			got := strings.Join(generatePrompt(tc.giveOpts), "\n")

			t.Log(got)

			for _, want := range tc.wantContains {
				if !strings.Contains(got, want) {
					t.Errorf("want %q to contain %q", got, want)
				}
			}

			for _, want := range tc.wantNot {
				if strings.Contains(got, want) {
					t.Errorf("want %q to not contain %q", got, want)
				}
			}
		})
	}
}
