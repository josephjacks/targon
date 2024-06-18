package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/ChainSafe/go-schnorrkel"
	"github.com/google/uuid"
	"github.com/nitishm/go-rejson/v4"
	"github.com/redis/go-redis/v9"
	"golang.org/x/crypto/sha3"
)

func safeEnv(env string) string {
	res, present := os.LookupEnv(env)
	if !present {
		log.Fatalf("Missing environment variable %s", env)
	}
	return res
}

func signMessage(message string, public string, private string) string {
	var pubk [32]byte
	data, err := hex.DecodeString(public)
	if err != nil {
		log.Fatalf("Failed to decode public key: %s", err)
	}
	copy(pubk[:], data)

	var prik [32]byte
	data, err = hex.DecodeString(private)
	if err != nil {
		log.Fatalf("Failed to decode private key: %s", err)
	}
	copy(prik[:], data)

	msg := []byte(message)
	priv := schnorrkel.SecretKey{}
	priv.Decode(prik)
	pub := schnorrkel.PublicKey{}
	pub.Decode(pubk)
	signingCtx := []byte("substrate")
	signingTranscript := schnorrkel.NewSigningContext(signingCtx, msg)
	sig, _ := priv.Sign(signingTranscript)
	sigEncode := sig.Encode()
	out := hex.EncodeToString(sigEncode[:])
	return "0x" + out
}

func hashString(str string) string {
	h := sha3.New256()
	h.Write([]byte(str))
	sum := h.Sum(nil)
	return hex.EncodeToString(sum)
}

func formatListToPythonString(list []string) string {
	strList := "["
	for i, element := range list {
		element = strconv.Quote(element)
		element = strings.TrimPrefix(element, "\"")
		element = strings.TrimSuffix(element, "\"")
		separator := "'"
		if strings.ContainsRune(element, '\'') && !strings.ContainsRune(element, '"') {
			separator = "\""
		} else {
			element = strings.ReplaceAll(element, "'", "\\'")
			element = strings.ReplaceAll(element, "\\\"", "\"")
		}
		if i != 0 {
			strList += ", "
		}
		strList += separator + element + separator
	}
	strList += "]"
	return strList
}

func sendEvent(c *Context, data map[string]any) {
	eventId := uuid.New().String()
	fmt.Fprintf(c.Response(), "id: %s\n", eventId)
	fmt.Fprintf(c.Response(), "event: new_message\n")
	eventData, _ := json.Marshal(data)
	fmt.Fprintf(c.Response(), "data: %s\n", string(eventData))
	fmt.Fprintf(c.Response(), "retry: %d\n\n", 1500)
	c.Response().Flush()
}

func buildPrompt(messages []RequestBodyMessages) string {
	prompt := ""
	for _, message := range messages {
		prompt += fmt.Sprintf("%s: %s\n", message.Role, message.Content)
	}
	return prompt
}

func queryMiners(c *Context, client *redis.Client, req RequestBody) {
	ctx := context.Background()
	defer ctx.Done()
	rh := rejson.NewReJSONHandler()
	rh.SetGoRedisClientWithContext(ctx, client)
	minerJSON, err := rh.JSONGet("miners", ".")
	if err != nil {
		c.Err.Printf("Failed to JSONGet: %s\n", err.Error())
		return
	}

	var minerOut []Miner
	err = json.Unmarshal(minerJSON.([]byte), &minerOut)
	if err != nil {
		c.Err.Printf("Failed to JSON Unmarshal: %s\n", err.Error())
		return
	}
	sources := []string{"https://google.com"}
	formattedSourcesList := formatListToPythonString(sources)
	prompt := buildPrompt(req.Messages)

	var hashes []string
	hashes = append(hashes, hashString(formattedSourcesList))
	hashes = append(hashes, hashString(prompt))
	bodyHash := hashString(strings.Join(hashes, ""))

	type Response struct {
		Res     *http.Response
		ColdKey string
		HotKey  string
	}

	response := make(chan Response)

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	var minerWaitGroup sync.WaitGroup
	minerWaitGroup.Add(len(minerOut))
	go func() {
		minerWaitGroup.Wait()
		close(response)
	}()
	tr := &http.Transport{
		MaxIdleConns:      10,
		IdleConnTimeout:   30 * time.Second,
		DisableKeepAlives: false,
	}
	httpClient := http.Client{Transport: tr}

	nonce := time.Now().UnixNano()
	for _, m := range minerOut {
		go func(miner Miner) {
			defer minerWaitGroup.Done()
			message := []string{fmt.Sprint(nonce), HOTKEY, miner.Hotkey, INSTANCE_UUID, bodyHash}
			joinedMessage := strings.Join(message, ".")
			signedMessage := signMessage(joinedMessage, PUBLIC_KEY, PRIVATE_KEY)
			port := fmt.Sprint(miner.Port)
			version := 672
			body := InferenceBody{
				Name:           "Inference",
				Timeout:        12.0,
				TotalSize:      0,
				HeaderSize:     0,
				RequiredFields: []string{"sources", "query", "seed"},
				Sources:        sources,
				Query:          prompt,
				BodyHash:       "",
				Dendrite: DendriteOrAxon{
					Ip:            "10.0.0.1",
					Version:       &version,
					Nonce:         &nonce,
					Uuid:          &INSTANCE_UUID,
					Hotkey:        HOTKEY,
					Signature:     &signedMessage,
					Port:          nil,
					StatusCode:    nil,
					StatusMessage: nil,
					ProcessTime:   nil,
				},
				Axon: DendriteOrAxon{
					StatusCode:    nil,
					StatusMessage: nil,
					ProcessTime:   nil,
					Version:       nil,
					Nonce:         nil,
					Uuid:          nil,
					Signature:     nil,
					Ip:            miner.Ip,
					Port:          &port,
					Hotkey:        miner.Hotkey,
				},
				SamplingParams: SamplingParams{
					Seed:                nil,
					Truncate:            nil,
					BestOf:              1,
					DecoderInputDetails: true,
					Details:             false,
					DoSample:            true,
					MaxNewTokens:        req.MaxTokens,
					RepetitionPenalty:   1.0,
					ReturnFullText:      false,
					Stop:                []string{"photographer"},
					Temperature:         .01,
					TopK:                10,
					TopNTokens:          5,
					TopP:                .9999999,
					TypicalP:            .9999999,
					Watermark:           false,
				},
				Completion: nil,
			}

			endpoint := "http://" + miner.Ip + ":" + fmt.Sprint(miner.Port) + "/Inference"
			out, err := json.Marshal(body)
			r, err := http.NewRequestWithContext(ctx, "POST", endpoint, bytes.NewBuffer(out))
			if err != nil {
				c.Warn.Printf("Failed miner request: %s\n", err.Error())
				return
			}
			r.Close = true
			r.Header["Content-Type"] = []string{"application/json"}
			r.Header["Connection"] = []string{"keep-alive"}
			r.Header["name"] = []string{"Inference"}
			r.Header["timeout"] = []string{"12.0"}
			r.Header["bt_header_axon_ip"] = []string{miner.Ip}
			r.Header["bt_header_axon_port"] = []string{strconv.Itoa(miner.Port)}
			r.Header["bt_header_axon_hotkey"] = []string{miner.Hotkey}
			r.Header["bt_header_dendrite_ip"] = []string{"10.0.0.1"}
			r.Header["bt_header_dendrite_version"] = []string{"672"}
			r.Header["bt_header_dendrite_nonce"] = []string{strconv.Itoa(int(nonce))}
			r.Header["bt_header_dendrite_uuid"] = []string{INSTANCE_UUID}
			r.Header["bt_header_dendrite_hotkey"] = []string{HOTKEY}
			r.Header["bt_header_input_obj_sources"] = []string{"W10="}
			r.Header["bt_header_input_obj_query"] = []string{"IiI="}
			r.Header["bt_header_dendrite_signature"] = []string{signedMessage}
			r.Header["header_size"] = []string{"0"}
			r.Header["total_size"] = []string{"0"}
			r.Header["computed_body_hash"] = []string{bodyHash}
			r.Header.Add("Accept-Encoding", "identity")
			res, err := httpClient.Do(r)
			if err != nil {
				c.Warn.Printf("Miner: %s %s\nError: %s\n", miner.Hotkey, miner.Coldkey, err.Error())
				if res != nil {
					res.Body.Close()
				}
				return
			}

			if res.StatusCode == http.StatusOK {
				bdy, _ := io.ReadAll(res.Body)
				res.Body.Close()
				c.Warn.Printf("Miner: %s %s\nError: %s\n", miner.Hotkey, miner.Coldkey, string(bdy))
				return
			}



			if res.StatusCode != http.StatusOK {
				bdy, _ := io.ReadAll(res.Body)
				res.Body.Close()
				c.Warn.Printf("Miner: %s %s\nError: %s\n", miner.Hotkey, miner.Coldkey, string(bdy))
				return
			}
			axon_version := res.Header.Get("Bt_header_axon_version")
			ver, err := strconv.Atoi(axon_version)
			if err != nil || ver < 672 {
				res.Body.Close()
				c.Warn.Printf("Miner: %s %s\nError: Axon version too low\n", miner.Hotkey, miner.Coldkey)
				return
			}
			response <- Response{Res: res, ColdKey: miner.Coldkey, HotKey: miner.Hotkey}
		}(m)
	}
	count := 0
	for {
		count++
		res, ok := <-response
		c.Info.Printf("Attempt: %d Miner: %s %s\n", count, res.HotKey, res.ColdKey)
		if !ok {
			return
		}
		reader := bufio.NewReader(res.Res.Body)
		finished := false
		ans := ""
		for {
			token, err := reader.ReadString(' ')
			if strings.Contains(token, "<s>") || strings.Contains(token, "</s>") || strings.Contains(token, "<im_end>") {
				finished = true
				token = strings.ReplaceAll(token, "<s>", "")
				token = strings.ReplaceAll(token, "</s>", "")
				token = strings.ReplaceAll(token, "<im_end>", "")
			}
			ans += token
			if err != nil && err != io.EOF {
				ans = ""
				c.Err.Println(err.Error())
				break
			}
			sendEvent(c, map[string]any{
				"type":     "answer",
				"text":     token,
				"finished": finished,
			})
			if err == io.EOF {
				break
			}
		}
		res.Res.Body.Close()
		if finished == false {
			continue
		}
		break
	}
	for {
		select {
		case res, ok := <-response:
			if !ok {
				response = nil
				break
			}
			res.Res.Body.Close()
		}
		if response == nil {
			break
		}
	}
}
