package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"
)

// OpenRouter is a provider for the OpenRouter API.
// It implements the Provider interface.
type OpenRouter struct {
	httpClient        httpClient
	apiKey, modelName string
}

// Ensure OpenRouter implements the Provider interface.
var _ Provider = (*OpenRouter)(nil)

type (
	// openrouterOptions holds custom configuration for the OpenRouter provider.
	openrouterOptions struct {
		httpClient httpClient
	}

	// OpenRouterOption allows customization of the OpenRouter provider.
	OpenRouterOption func(*openrouterOptions)
)

// WithOpenRouterHttpClient sets a custom HTTP client for the OpenRouter provider.
func WithOpenRouterHttpClient(c httpClient) OpenRouterOption {
	return func(o *openrouterOptions) { o.httpClient = c }
}

// NewOpenRouter creates a new OpenRouter provider.
func NewOpenRouter(apiKey, model string, opts ...OpenRouterOption) *OpenRouter {
	var options openrouterOptions
	for _, o := range opts {
		o(&options)
	}

	p := OpenRouter{
		httpClient: options.httpClient,
		apiKey:     apiKey,
		modelName:  model,
	}

	if p.httpClient == nil {
		p.httpClient = &http.Client{
			Timeout:   60 * time.Second,
			Transport: &http.Transport{ForceAttemptHTTP2: true},
		}
	}

	return &p
}

// Query sends a request to the OpenRouter API to generate a commit message
// based on the provided diff and commit history.
func (p *OpenRouter) Query(ctx context.Context, changes, commits string, opts ...Option) (*Response, error) {
	opt := options{}.Apply(opts...)
	instructions := GeneratePrompt(opts...)

	if opt.MaxOutputTokens == 0 {
		opt.MaxOutputTokens = defaultMaxOutputTokens // set default value
	}

	req, err := p.newRequest(ctx, instructions, changes, commits, opt)
	if err != nil {
		return nil, err
	}

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected OpenRouter API response status code: %d", resp.StatusCode)
	}

	var result struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	if len(result.Choices) == 0 || result.Choices[0].Message.Content == "" {
		return nil, fmt.Errorf("no response from OpenRouter API")
	}

	if opt.ShortMessageOnly {
		parts := strings.Split(result.Choices[0].Message.Content, "\n")
		if len(parts) == 0 {
			return nil, fmt.Errorf("no response from OpenRouter API")
		}
		return &Response{Prompt: instructions, Answer: parts[0]}, nil
	}

	return &Response{Prompt: instructions, Answer: result.Choices[0].Message.Content}, nil
}

// newRequest creates a new HTTP request for the OpenRouter API.
// The request payload is similar to the OpenAI chat completions endpoint.
func (p *OpenRouter) newRequest(ctx context.Context, instructions, changes, commits string, o options) (*http.Request, error) {
	type message struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}

	requestBody := struct {
		Model       string    `json:"model"`
		Messages    []message `json:"messages"`
		Temperature float64   `json:"temperature"`
		TopP        float64   `json:"top_p"`
		N           int       `json:"n"`
		MaxTokens   int64     `json:"max_tokens"`
		Stream      bool      `json:"stream"`
	}{
		Model:       p.modelName,
		Temperature: 0.1,
		TopP:        0.1,
		N:           1,
		MaxTokens:   o.MaxOutputTokens,
		Stream:      false,
		Messages: []message{
			{Role: "system", Content: instructions},
			{Role: "user", Content: wrapChanges(changes)},
			{Role: "user", Content: wrapCommits(commits)},
		},
	}

	body, err := json.Marshal(requestBody)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://openrouter.ai/api/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}

	// Required headers
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", p.apiKey))

	// Optional headers for OpenRouter tracking
	req.Header.Set("HTTP-Referer", "https://github.com/tarampampam/describe-commit")
	req.Header.Set("X-Title", "describe-commit")
	req.Header.Set("X-Version", "1.0.0")

	return req, nil
}
