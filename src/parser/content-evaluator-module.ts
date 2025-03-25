import Decimal from "decimal.js";
import { CommentAssociation, commentEnum, CommentKind, CommentType } from "../configuration/comment-types";
import { ContentEvaluatorConfiguration } from "../configuration/content-evaluator-config";
import { IssueActivity } from "../issue-activity";
import { AllComments, CommentToEvaluate, PrCommentToEvaluate } from "../types/content-evaluator-module-type";
import { BaseModule } from "../types/module";
import { ContextPlugin } from "../types/plugin-input";
import { GithubCommentScore, Result } from "../types/results";
import { encodingForModel } from "js-tiktoken";

/**
 * Evaluates and rates comments.
 */
export class ContentEvaluatorModule extends BaseModule {
  readonly _configuration: ContentEvaluatorConfiguration | null = this.context.config.incentives.contentEvaluator;
  private readonly _fixedRelevances: { [k: string]: number } = {};
  private _tokenLimit: number = 0;

  _getEnumValue(key: CommentType) {
    let res = 0;

    key.split("_").forEach((value) => {
      res |= Number(commentEnum[value as keyof typeof commentEnum]);
    });
    return res;
  }

  constructor(context: ContextPlugin) {
    super(context);
    if (this._configuration?.multipliers) {
      this._fixedRelevances = this._configuration.multipliers.reduce((acc, curr) => {
        return {
          ...acc,
          [curr.role.reduce((a, b) => this._getEnumValue(b) | a, 0)]: curr.relevance,
        };
      }, {});
    }
  }

  get enabled(): boolean {
    if (!this._configuration) {
      this.context.logger.warn(
        "The configuration for the module ContentEvaluatorModule is invalid or missing, disabling."
      );
      return false;
    }
    return true;
  }

  async transform(data: Readonly<IssueActivity>, result: Result) {
    if (!this._configuration?.openAi.tokenCountLimit) {
      throw this.context.logger.fatal("Token count limit is missing, comments cannot be evaluated.");
    }
    const promises: Promise<GithubCommentScore[]>[] = [];
    const allComments: { id: number; comment: string; author: string }[] = [];

    for (const [user, data] of Object.entries(result)) {
      if (data.comments?.length) {
        allComments.push(
          ...data.comments.map((comment) => ({ id: comment.id, comment: comment.content, author: user }))
        );
      }
    }

    for (const key of Object.keys(result)) {
      const currentElement = result[key];
      const comments = currentElement.comments ?? [];
      const specificationBody = data.self?.body;

      if (specificationBody && comments.length) {
        promises.push(
          this._processComment(comments).then((commentsWithScore) => (currentElement.comments = commentsWithScore))
        );
      }
    }

    await Promise.all(promises);
    return result;
  }

  async _processComment(comments: Readonly<GithubCommentScore>[]) {
    const commentsWithScore: GithubCommentScore[] = [...comments];
    const { commentsToEvaluate, prCommentsToEvaluate } = this._splitCommentsByPrompt(commentsWithScore);

    const relevancesByAi: { [k: string]: number } = {};

    commentsToEvaluate.forEach((comment) => {
      relevancesByAi[`${comment.id}`] = 0.8;
    });
    prCommentsToEvaluate.forEach((comment) => {
      relevancesByAi[`${comment.id}`] = 0.7;
    });

    for (const currentComment of commentsWithScore) {
      let currentRelevance = 1; // For comments not in fixed relevance types and missed by OpenAI evaluation
      if (this._fixedRelevances[currentComment.commentType]) {
        currentRelevance = this._fixedRelevances[currentComment.commentType];
      } else if (!isNaN(relevancesByAi[currentComment.id])) {
        currentRelevance = relevancesByAi[currentComment.id];
      }

      const currentReward = new Decimal(currentComment.score?.reward ?? 0);
      const priority =
        // We do not apply priority multiplier on issue specification
        currentComment.score?.priority && !(currentComment.commentType & CommentAssociation.SPECIFICATION)
          ? currentComment.score.priority
          : 1;

      currentComment.score = {
        ...(currentComment.score || { multiplier: 0 }),
        relevance: new Decimal(currentRelevance).toNumber(),
        priority: priority,
        reward: currentReward.mul(currentRelevance).mul(priority).toDecimalPlaces(3).toNumber(),
      };
    }

    return commentsWithScore;
  }

  /**
   * Will try to predict the maximum of tokens expected, to a maximum of totalTokenLimit.
   */
  _calculateMaxTokens(prompt: string, totalTokenLimit: number = 16384) {
    const tokenizer = encodingForModel("gpt-4o");
    const inputTokens = tokenizer.encode(prompt).length * 2; // Safety margin
    return Math.min(inputTokens, totalTokenLimit);
  }

  _generateDummyResponse(comments: { id: number; comment: string }[]) {
    return comments.reduce<Record<string, number>>((acc, curr) => {
      return { ...acc, [curr.id]: 0.5 };
    }, {});
  }

  _splitCommentsByPrompt(commentsWithScore: Readonly<GithubCommentScore>[]): {
    commentsToEvaluate: CommentToEvaluate[];
    prCommentsToEvaluate: PrCommentToEvaluate[];
  } {
    const commentsToEvaluate: CommentToEvaluate[] = [];
    const prCommentsToEvaluate: PrCommentToEvaluate[] = [];
    for (const currentComment of commentsWithScore) {
      if (!this._fixedRelevances[currentComment.commentType]) {
        if (currentComment.commentType & CommentKind.PULL) {
          prCommentsToEvaluate.push({
            id: currentComment.id,
            comment: currentComment.content,
            diffHunk: currentComment?.diffHunk,
          });
        } else {
          commentsToEvaluate.push({
            id: currentComment.id,
            comment: currentComment.content,
          });
        }
      }
    }
    return { commentsToEvaluate, prCommentsToEvaluate };
  }

  _splitArrayToChunks<T extends CommentToEvaluate[] | AllComments>(array: T, chunks: number) {
    const arrayCopy = [...array];
    const result = [];
    for (let i = chunks; i > 0; i--) {
      result.push(arrayCopy.splice(0, Math.ceil(arrayCopy.length / i)));
    }
    return result;
  }

  _generatePromptForComments(issue: string, userComments: CommentToEvaluate[], allComments: AllComments) {
    if (!issue?.length) {
      throw new Error("Issue specification comment is missing or empty");
    }
    const allCommentsMap = allComments.map((value) => `${value.id} - ${value.author}: "${value.comment}"`);
    const userCommentsMap = userComments.map((value) => `${value.id}: "${value.comment}"`);
    return `
      Evaluate the relevance of GitHub comments to an issue. Provide a raw JSON object with comment IDs and their relevance scores.
      Issue: ${issue}

      All comments:
      ${allCommentsMap.join("\n")}

      Comments to evaluate:
      ${userCommentsMap.join("\n")}

      Instructions:
      1. Read all comments carefully, considering their context and content.
      2. Evaluate each comment in the "Comments to evaluate" section.
      3. Assign a relevance score from 0 to 1 for each comment:
        - 0: Not related (e.g., spam)
        - 1: Highly relevant (e.g., solutions, bug reports)
      4. Consider:
        - Relation to the issue description
        - Connection to other comments
        - Contribution to issue resolution
      5. Handle GitHub-flavored markdown:
        - Ignore text beginning with '>' as it references another comment
        - Distinguish between referenced text and the commenter's own words
        - Only evaluate the relevance of the commenter's original content
      6. Return only a JSON object: {ID: score}

      Notes:
      - Even minor details may be significant.
      - Comments may reference earlier comments.
      - The number of entries in the JSON response must equal ${userCommentsMap.length}.

      IMPORTANT: Do not use markdown formatting. Do not include backticks or code blocks. Return just the plain JSON object text that can be directly parsed.
    `;
  }

  _generatePromptForPrComments(issue: string, userComments: PrCommentToEvaluate[]) {
    if (!issue?.length) {
      throw new Error("Issue specification comment is missing or empty");
    }
    return `I need to evaluate the value of a GitHub contributor's comments in a pull request. 
    Some of these comments are code review comments, and some are general suggestions or a part of the discussion. 
    I'm interested in how much each comment helps to solve the GitHub issue and improve code quality. 
    Please provide a float between 0 and 1 to represent the value of each comment. 
    A score of 1 indicates that the comment is very valuable and significantly improves the submitted solution and code quality, whereas a score of 0 indicates a negative or zero impact. 
    A stringified JSON is given below that contains the specification of the GitHub issue, and comments by different contributors. 
    The property "diffHunk" presents the chunk of code being addressed for a possible change in a code review comment. 
    
    \`\`\`
    ${JSON.stringify({ specification: issue, comments: userComments })}
    \`\`\`\
  
    To what degree are each of the comments valuable? 
    Please reply with ONLY a raw JSON object where each key is the comment ID given in JSON above, and the value is a float number between 0 and 1 corresponding to the comment. 
    The float number should represent the value of the comment for improving the issue solution and code quality. The total number of properties in your JSON response should equal exactly ${userComments.length}.
    
    IMPORTANT: Do not use markdown formatting. Do not include backticks or code blocks. Return just the plain JSON object text that can be directly parsed.`;
  }
}
