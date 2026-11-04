import type { ChatAssistantMessage, ChatToolMessage } from '../../../types/chat'
import type { ToolCallResponse } from '../../../types/tool-call.types'
import { ToolCallResponseStatus } from '../../../types/tool-call.types'
import {
  ASK_USER_QUESTION_OTHER_ID,
  type AskUserQuestionArgs,
  type AskUserQuestionItem,
  type AskUserQuestionOption,
} from '../../mcp/localFileTools'
import { createCliToolCallRequest } from '../tool-call'
import type { CliRuntimeId } from '../types'

import { toolPair } from './mapping'
import type { PiRpcRecord } from './transport'

/**
 * pi's extension UI sub-protocol (`docs/rpc-extension-ui.md`) is the only way
 * a pi extension can ask the human anything: an extension that needs a
 * permission grant or a decision calls `ctx.ui.confirm` / `select` / `input` /
 * `editor`, and pi turns that call into one `extension_ui_request` record on
 * stdout that blocks the extension until we answer on stdin. pi core has no
 * approval gate of its own and no builtin ask-user tool, so *every* "the agent
 * is waiting for you" moment in a pi session arrives here — an approval plugin,
 * a destructive-command guard, a workflow extension asking which draft to use.
 * Answering generically is therefore what adapts the runtime to extensions we
 * have never seen, rather than special-casing the handful we know.
 *
 * Only the dialog methods become cards. `notify` is surfaced as a notice; the
 * other fire-and-forget methods (`setStatus`, `setWidget`, `setTitle`,
 * `set_editor_text`) describe pi's own TUI chrome, which has no counterpart
 * here.
 */

export type PiExtensionUiDialogMethod =
  | 'select'
  | 'confirm'
  | 'input'
  | 'editor'

export type PiExtensionUiDialog = {
  /** pi's request id; also the card's tool call id, so an answer routes back. */
  id: string
  method: PiExtensionUiDialogMethod
  title: string
  /** `select` only: the exact strings pi compares the answer against. */
  options: readonly string[]
  /** `confirm` only: the body pi shows under the title. */
  message: string
  /** `input` placeholder / `editor` prefill. */
  prefill: string | null
  /** pi resolves the dialog itself when this elapses. */
  timeoutMs: number | null
}

/**
 * The one question a dialog becomes. pi asks exactly one thing per dialog, so
 * the card, its schema-validated question and the answer mapping travel
 * together.
 */
export type PiExtensionUiQuestion = {
  question: AskUserQuestionItem
  /** option id -> the exact string pi hands back to the extension. */
  optionValues: Readonly<Record<string, string>>
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === 'object' && !Array.isArray(value)

const getString = (value: unknown): string | null =>
  typeof value === 'string' && value.trim() ? value.trim() : null

/** Content-preserving counterpart of `getString`: an editor prefill is code. */
const getRawString = (value: unknown): string | null =>
  typeof value === 'string' && value.trim() ? value : null

const asTimeoutMs = (value: unknown): number | null =>
  typeof value === 'number' && Number.isFinite(value) && value > 0
    ? value
    : null

export const parsePiExtensionUiDialog = (
  record: PiRpcRecord,
): PiExtensionUiDialog | null => {
  if (getString(record.type) !== 'extension_ui_request') return null
  const id = getString(record.id)
  const method = getString(record.method)
  if (!id || !method) return null
  const base = {
    id,
    title: getString(record.title) ?? '',
    options: [] as string[],
    message: '',
    prefill: null as string | null,
    timeoutMs: asTimeoutMs(record.timeout),
  }
  switch (method) {
    case 'select':
      return {
        ...base,
        method,
        options: Array.isArray(record.options)
          ? record.options.filter(
              (option): option is string => typeof option === 'string',
            )
          : [],
      }
    case 'confirm':
      return { ...base, method, message: getString(record.message) ?? '' }
    case 'input':
      return { ...base, method, prefill: getString(record.placeholder) }
    case 'editor':
      return {
        ...base,
        method,
        // pi's editor dialog has no timeout: it waits for the human.
        timeoutMs: null,
        prefill: getRawString(record.prefill),
      }
    default:
      return null
  }
}

/** The `notify` method: user-visible text, no answer expected. */
export const parsePiExtensionUiNotification = (
  record: PiRpcRecord,
): string | null => {
  if (getString(record.type) !== 'extension_ui_request') return null
  if (getString(record.method) !== 'notify') return null
  return getString(record.message)
}

/**
 * The panel's schema requires a non-empty prompt and non-empty, unique option
 * ids that avoid the reserved "Other" id; pi validates none of that. Ids are
 * therefore derived here and the original strings are kept beside them, so the
 * answer pi receives is byte-for-byte what the extension offered.
 */
const QUESTION_ID = 'extension-ui'

const uniqueOptionId = (
  label: string,
  index: number,
  used: Set<string>,
): string => {
  const base =
    label && label !== ASK_USER_QUESTION_OTHER_ID
      ? label
      : `option-${index + 1}`
  if (!used.has(base)) return base
  let suffix = 2
  while (used.has(`${base}-${suffix}`)) suffix += 1
  return `${base}-${suffix}`
}

/**
 * A `single_select` needs at least two options, so a dialog that offers fewer
 * degrades to free text with the offered values spelled out in the prompt —
 * the answer is still sent back verbatim, and nothing the extension said is
 * lost on the way.
 */
const buildSelectQuestion = (
  dialog: PiExtensionUiDialog,
  prompt: string,
): PiExtensionUiQuestion => {
  const used = new Set<string>()
  const optionValues: Record<string, string> = {}
  const options: AskUserQuestionOption[] = []
  dialog.options.forEach((raw, index) => {
    const label = raw.trim()
    if (!label) return
    const id = uniqueOptionId(label, index, used)
    used.add(id)
    optionValues[id] = raw
    options.push({ id, label })
  })
  if (options.length >= 2) {
    return {
      question: {
        id: QUESTION_ID,
        prompt,
        inputType: 'single_select',
        options,
      },
      optionValues,
    }
  }
  const listed = dialog.options.filter((option) => option.trim())
  return {
    question: {
      id: QUESTION_ID,
      prompt: listed.length > 0 ? `${prompt}\n\n${listed.join('\n')}` : prompt,
      inputType: 'free_text',
    },
    optionValues: {},
  }
}

export const buildPiExtensionUiQuestion = (
  dialog: PiExtensionUiDialog,
): PiExtensionUiQuestion => {
  const title = dialog.title || dialog.method
  if (dialog.method === 'select') return buildSelectQuestion(dialog, title)
  // The ask-user panel has no prefill/placeholder slot, so pi's pre-filled text
  // rides along in the prompt instead of being dropped.
  const prompt = dialog.prefill ? `${title}\n\n${dialog.prefill}` : title
  return {
    question: { id: QUESTION_ID, prompt, inputType: 'free_text' },
    optionValues: {},
  }
}

const dialogInput = (dialog: PiExtensionUiDialog): Record<string, unknown> => ({
  title: dialog.title,
  ...(dialog.options.length > 0 ? { options: dialog.options } : {}),
  ...(dialog.message ? { message: dialog.message } : {}),
  ...(dialog.prefill ? { prefill: dialog.prefill } : {}),
})

/** `confirm` is a yes/no decision, so it keeps the host's approval card. */
export const buildPiExtensionUiApprovalMessages = (
  dialog: PiExtensionUiDialog,
  runtimeId: CliRuntimeId,
): [ChatAssistantMessage, ChatToolMessage] =>
  toolPair(
    createCliToolCallRequest({
      id: dialog.id,
      input: dialogInput(dialog),
      metadata: {
        runtimeId,
        eventType: 'extension_ui_request',
        name: dialog.method,
        capability: 'permission_request',
      },
    }),
    { status: ToolCallResponseStatus.PendingApproval },
  )

export const buildPiExtensionUiQuestionMessages = (
  dialog: PiExtensionUiDialog,
  question: PiExtensionUiQuestion,
  runtimeId: CliRuntimeId,
): [ChatAssistantMessage, ChatToolMessage] => {
  const presentationArguments: AskUserQuestionArgs = {
    questions: [question.question],
  }
  return toolPair(
    createCliToolCallRequest({
      id: dialog.id,
      input: dialogInput(dialog),
      metadata: {
        runtimeId,
        eventType: 'extension_ui_request',
        name: dialog.method,
        capability: 'user_question',
        // The ask-user panel reads the schema-validated questions from here
        // rather than from the raw dialog the extension sent.
        presentationArguments,
      },
    }),
    { status: ToolCallResponseStatus.AwaitingUserInput },
  )
}

/** Re-publishes a card that settled without a host action (timeout). */
export const settlePiExtensionUiCard = (
  card: ChatToolMessage,
  response: ToolCallResponse,
): ChatToolMessage => ({
  ...card,
  toolCalls: card.toolCalls.map((call) => ({ ...call, response })),
})

export const toPiExtensionUiCancelResponse = (
  id: string,
): Record<string, unknown> => ({
  type: 'extension_ui_response',
  id,
  cancelled: true,
})

export type PiExtensionUiAnswer =
  | {
      ok: true
      /** What pi is told the human answered. */
      value: string
      /** The settled card content, in the ask-user panel's answer shape. */
      settled: ToolCallResponse
    }
  | { ok: false }

const findSubmittedAnswer = (
  answer: unknown,
  questionId: string,
): Record<string, unknown> | null => {
  if (!isRecord(answer)) return null
  if (getString(answer.type) !== 'user_answers') return null
  const answers = answer.answers
  if (!Array.isArray(answers)) return null
  const match = answers.find(
    (entry) => isRecord(entry) && entry.id === questionId,
  )
  return match ? (match as Record<string, unknown>) : null
}

const settledAnswer = (
  question: PiExtensionUiQuestion,
  submitted: Record<string, unknown>,
): ToolCallResponse => ({
  status: ToolCallResponseStatus.Success,
  data: {
    type: 'text',
    // Same envelope the native ask_user_question tool returns, so the panel
    // renders the answer it already knows how to display.
    text: JSON.stringify({
      type: 'user_answers',
      answers: [
        {
          id: question.question.id,
          question: question.question.prompt,
          inputType: question.question.inputType,
          ...submitted,
        },
      ],
    }),
  },
})

/**
 * The panel injects its own "Other" option into every single_select, so a
 * typed answer is the one case where an offered string leaves pi's fixed list.
 * Extensions that switch on the exact option string — the usual `['Yes','No']`
 * permission prompt — would read a typed "yes" as neither, so a typed answer
 * that spells an offered option is sent as that option.
 */
const findOfferedOption = (
  question: PiExtensionUiQuestion,
  text: string,
): { id: string; value: string } | null => {
  const needle = text.toLowerCase()
  if (!needle) return null
  for (const [id, value] of Object.entries(question.optionValues)) {
    if (value.trim().toLowerCase() === needle) return { id, value }
  }
  return null
}

/**
 * Turns the ask-user panel's submit payload into the string pi's dialog
 * expects. `ok: false` means the payload did not answer *this* question — the
 * caller treats it as a cancellation rather than sending pi something it would
 * misinterpret.
 */
export const resolvePiExtensionUiAnswer = (
  question: PiExtensionUiQuestion,
  answer: unknown,
): PiExtensionUiAnswer => {
  const submitted = findSubmittedAnswer(answer, question.question.id)
  if (!submitted) return { ok: false }
  if (question.question.inputType === 'free_text') {
    if (typeof submitted.value !== 'string') return { ok: false }
    return {
      ok: true,
      value: submitted.value,
      settled: settledAnswer(question, { value: submitted.value }),
    }
  }
  const optionId = getString(submitted.value)
  if (!optionId) return { ok: false }
  if (optionId === ASK_USER_QUESTION_OTHER_ID) {
    const otherText =
      typeof submitted.otherText === 'string' ? submitted.otherText.trim() : ''
    const offered = findOfferedOption(question, otherText)
    if (offered) {
      return {
        ok: true,
        value: offered.value,
        settled: settledAnswer(question, { value: offered.id }),
      }
    }
    // A genuinely custom answer: it is the whole answer, empty or not — a
    // select answer is never picked at random on the human's behalf.
    return {
      ok: true,
      value: otherText,
      settled: settledAnswer(question, {
        value: ASK_USER_QUESTION_OTHER_ID,
        otherText,
      }),
    }
  }
  return {
    ok: true,
    value: question.optionValues[optionId] ?? optionId,
    settled: settledAnswer(question, { value: optionId }),
  }
}
