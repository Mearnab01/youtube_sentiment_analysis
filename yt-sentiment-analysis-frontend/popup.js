// popup.js

document.addEventListener("DOMContentLoaded", async () => {
  const outputDiv = document.getElementById("output");
  const API_KEY = 'AIzaSyDAdRpmLSD0SdaX_1shixX5TGoN-qTftIM';
  const API_URL = 'http://localhost:8080'; // FIXED: was a dead remote IP with a trailing slash

  /* --------------------------------------------------------------------
   * Presentation helpers (markup only — no business logic here)
   * ------------------------------------------------------------------ */

  const icons = {
    warning: `<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 9v4m0 4h.01M10.29 3.86l-8.18 14.16A1.5 1.5 0 0 0 3.4 20.5h17.2a1.5 1.5 0 0 0 1.3-2.48L13.71 3.86a1.5 1.5 0 0 0-2.42 0z" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
    inbox: `<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M4 12h4l2 3h4l2-3h4M4 12l1.5-6.5A2 2 0 0 1 7.44 4h9.12a2 2 0 0 1 1.94 1.5L20 12M4 12v6a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-6" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
    link: `<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M9 12h6M8 7h8a4 4 0 1 1 0 8h-2M16 17H8a4 4 0 1 1 0-8h2" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>`
  };

  function statusLine(message) {
    return `<div class="status-line"><span class="spinner"></span><span>${message}</span></div>`;
  }

  function errorAlert(message) {
    return `<div class="alert"><span class="alert__icon">${icons.warning}</span><span>${message}</span></div>`;
  }

  function statePanel(icon, title, desc) {
    return `
      <div class="state-panel">
        <div class="state-panel__icon">${icon}</div>
        <div class="state-panel__title">${title}</div>
        <div class="state-panel__desc">${desc}</div>
      </div>`;
  }

  function sentimentMeta(rawValue) {
    const value = parseInt(rawValue);
    if (value > 0) return { label: "Positive", modifier: "positive" };
    if (value < 0) return { label: "Negative", modifier: "negative" };
    return { label: "Neutral", modifier: "neutral" };
  }

  function sectionHeader(label) {
    return `<div class="section-title"><span class="section-title__dot"></span>${label}</div>`;
  }

  function imageLoadingPlaceholder(label) {
    return `<div class="image-card__loading"><span class="spinner"></span><span>${label}</span></div>`;
  }

  // Get the current tab's URL
  chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
    const url = tabs[0].url;
    const youtubeRegex = /^https:\/\/(?:www\.)?youtube\.com\/watch\?v=([\w-]{11})/;
    const match = url.match(youtubeRegex);

    if (match && match[1]) {
      const videoId = match[1];
      outputDiv.innerHTML = `
        <div class="video-id-row">
          <span class="video-id-row__label">Video ID</span>
          <span class="video-id-row__value">${videoId}</span>
        </div>
        <div id="status-area">${statusLine("Fetching comments&hellip;")}</div>
      `;

      const comments = await fetchComments(videoId);
      if (comments.length === 0) {
        document.getElementById("status-area").innerHTML = statePanel(
          icons.inbox,
          "No comments found",
          "This video doesn't have any comments to analyze yet."
        );
        return;
      }

      document.getElementById("status-area").innerHTML = statusLine(
        `Analyzing ${comments.length} comments&hellip;`
      );
      const predictions = await getSentimentPredictions(comments);

      if (predictions) {
        document.getElementById("status-area").innerHTML = "";

        const sentimentCounts = { "1": 0, "0": 0, "-1": 0 };
        const sentimentData = [];
        const totalSentimentScore = predictions.reduce((sum, item) => sum + parseInt(item.sentiment), 0);
        predictions.forEach((item, index) => {
          sentimentCounts[item.sentiment]++;
          sentimentData.push({
            timestamp: item.timestamp,
            sentiment: parseInt(item.sentiment)
          });
        });

        const totalComments = comments.length;
        const uniqueCommenters = new Set(comments.map(comment => comment.authorId)).size;
        const totalWords = comments.reduce((sum, comment) => sum + comment.text.split(/\s+/).filter(word => word.length > 0).length, 0);
        const avgWordLength = (totalWords / totalComments).toFixed(2);
        const avgSentimentScore = (totalSentimentScore / totalComments).toFixed(2);
        const normalizedSentimentScore = (((parseFloat(avgSentimentScore) + 1) / 2) * 10).toFixed(2);

        outputDiv.innerHTML += `
          <div class="section">
            ${sectionHeader("Comment Analysis Summary")}
            <div class="metrics-container">
              <div class="metric">
                <div class="metric-title">Total Comments</div>
                <div class="metric-value">${totalComments}</div>
              </div>
              <div class="metric">
                <div class="metric-title">Unique Commenters</div>
                <div class="metric-value">${uniqueCommenters}</div>
              </div>
              <div class="metric">
                <div class="metric-title">Avg Comment Length</div>
                <div class="metric-value">${avgWordLength}<span style="font-size:12px;color:var(--text-faint)"> words</span></div>
              </div>
              <div class="metric">
                <div class="metric-title">Avg Sentiment Score</div>
                <div class="metric-value">${normalizedSentimentScore}<span style="font-size:12px;color:var(--text-faint)">/10</span></div>
              </div>
            </div>
          </div>
        `;

        outputDiv.innerHTML += `
          <div class="section">
            ${sectionHeader("Sentiment Distribution")}
            <div class="card" style="padding:0;">
              <div id="chart-container" class="image-card">${imageLoadingPlaceholder("Rendering chart&hellip;")}</div>
            </div>
          </div>`;
        await fetchAndDisplayChart(sentimentCounts);

        outputDiv.innerHTML += `
          <div class="section">
            ${sectionHeader("Sentiment Trend Over Time")}
            <div class="card" style="padding:0;">
              <div id="trend-graph-container" class="image-card">${imageLoadingPlaceholder("Rendering trend&hellip;")}</div>
            </div>
          </div>`;
        await fetchAndDisplayTrendGraph(sentimentData);

        outputDiv.innerHTML += `
          <div class="section">
            ${sectionHeader("Comment Wordcloud")}
            <div class="card" style="padding:0;">
              <div id="wordcloud-container" class="image-card">${imageLoadingPlaceholder("Building wordcloud&hellip;")}</div>
            </div>
          </div>`;
        await fetchAndDisplayWordCloud(comments.map(comment => comment.text));

        outputDiv.innerHTML += `
          <div class="section">
            ${sectionHeader("Top 25 Comments")}
            <ul class="comment-list">
              ${predictions.slice(0, 25).map((item, index) => {
                const meta = sentimentMeta(item.sentiment);
                return `
                <li class="comment-item">
                  <span class="comment-item__index">${index + 1}</span>
                  <div class="comment-item__body">
                    <div class="comment-item__text">${item.comment}</div>
                    <span class="comment-sentiment comment-sentiment--${meta.modifier}">${meta.label}</span>
                  </div>
                </li>`;
              }).join('')}
            </ul>
          </div>`;
      }
    } else {
      outputDiv.innerHTML = statePanel(
        icons.link,
        "Not a YouTube video",
        "Open a youtube.com/watch page and reopen this popup to analyze comments."
      );
    }
  });

  async function fetchComments(videoId) {
    let comments = [];
    let pageToken = "";
    try {
      while (comments.length < 500) {
        const response = await fetch(`https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId=${videoId}&maxResults=100&pageToken=${pageToken}&key=${API_KEY}`);
        const data = await response.json();
        if (data.items) {
          data.items.forEach(item => {
            const commentText = item.snippet.topLevelComment.snippet.textOriginal;
            const timestamp = item.snippet.topLevelComment.snippet.publishedAt;
            const authorId = item.snippet.topLevelComment.snippet.authorChannelId?.value || 'Unknown';
            comments.push({ text: commentText, timestamp: timestamp, authorId: authorId });
          });
        }
        pageToken = data.nextPageToken;
        if (!pageToken) break;
      }
    } catch (error) {
      console.error("Error fetching comments:", error);
      const statusArea = document.getElementById("status-area");
      if (statusArea) statusArea.innerHTML = errorAlert("Error fetching comments.");
    }
    return comments;
  }

  async function getSentimentPredictions(comments) {
    try {
      const validComments = comments.filter(c => c.text && c.text.trim().length > 0);

      const response = await fetch(`${API_URL}/predict_with_timestamps`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ comments: validComments })
      });
      const result = await response.json();
      if (response.ok) {
        return result;
      } else {
        throw new Error(result.error || 'Error fetching predictions');
      }
    } catch (error) {
      console.error("Error fetching predictions:", error);
      const statusArea = document.getElementById("status-area");
      if (statusArea) statusArea.innerHTML = errorAlert("Error fetching sentiment predictions.");
      return null;
    }
  }

  async function fetchAndDisplayChart(sentimentCounts) {
    const chartContainer = document.getElementById('chart-container');
    try {
      const response = await fetch(`${API_URL}/generate_chart`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sentiment_counts: sentimentCounts })
      });
      if (!response.ok) {
        throw new Error('Failed to fetch chart image');
      }
      const blob = await response.blob();
      const imgURL = URL.createObjectURL(blob);
      const img = document.createElement('img');
      img.src = imgURL;
      img.alt = "Sentiment distribution chart";
      chartContainer.innerHTML = "";
      chartContainer.appendChild(img);
    } catch (error) {
      console.error("Error fetching chart image:", error);
      chartContainer.innerHTML = errorAlert("Error fetching chart image.");
    }
  }

  async function fetchAndDisplayWordCloud(comments) {
    const wordcloudContainer = document.getElementById('wordcloud-container');
    try {
      const response = await fetch(`${API_URL}/generate_wordcloud`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ comments })
      });
      if (!response.ok) {
        throw new Error('Failed to fetch word cloud image');
      }
      const blob = await response.blob();
      const imgURL = URL.createObjectURL(blob);
      const img = document.createElement('img');
      img.src = imgURL;
      img.alt = "Comment wordcloud";
      wordcloudContainer.innerHTML = "";
      wordcloudContainer.appendChild(img);
    } catch (error) {
      console.error("Error fetching word cloud image:", error);
      wordcloudContainer.innerHTML = errorAlert("Error fetching word cloud image.");
    }
  }

  async function fetchAndDisplayTrendGraph(sentimentData) {
    const trendGraphContainer = document.getElementById('trend-graph-container');
    try {
      const response = await fetch(`${API_URL}/generate_trend_graph`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sentiment_data: sentimentData })
      });
      if (!response.ok) {
        throw new Error('Failed to fetch trend graph image');
      }
      const blob = await response.blob();
      const imgURL = URL.createObjectURL(blob);
      const img = document.createElement('img');
      img.src = imgURL;
      img.alt = "Sentiment trend over time";
      trendGraphContainer.innerHTML = "";
      trendGraphContainer.appendChild(img);
    } catch (error) {
      console.error("Error fetching trend graph image:", error);
      trendGraphContainer.innerHTML = errorAlert("Error fetching trend graph image.");
    }
  }
});