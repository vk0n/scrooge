FROM node:20-bookworm-slim AS builder

WORKDIR /app/frontend

ARG INTERNAL_API_BASE_URL=http://api:8000
ARG NEXT_PUBLIC_DISPLAY_TIMEZONE=Europe/Kyiv
ENV INTERNAL_API_BASE_URL=${INTERNAL_API_BASE_URL}
ENV NEXT_PUBLIC_DISPLAY_TIMEZONE=${NEXT_PUBLIC_DISPLAY_TIMEZONE}

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend ./
RUN npm run build


FROM node:20-bookworm-slim

ENV NODE_ENV=production

WORKDIR /app/frontend

COPY --from=builder /app/frontend/package.json /app/frontend/package-lock.json ./
RUN npm ci --omit=dev

COPY --from=builder /app/frontend/.next /app/frontend/.next
COPY --from=builder /app/frontend/next.config.mjs /app/frontend/next.config.mjs
COPY --from=builder /app/frontend/public /app/frontend/public

EXPOSE 3000

CMD ["npm", "run", "start", "--", "-p", "3000"]
