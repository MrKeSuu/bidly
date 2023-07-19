build-debug:
	export LEGACY_NDK=~/.buildozer/android/platform/android-ndk-r21e
	buildozer -v android debug

deploy-log:
	export LEGACY_NDK=~/.buildozer/android/platform/android-ndk-r21e
	buildozer android deploy run logcat

complete:
	export LEGACY_NDK=~/.buildozer/android/platform/android-ndk-r21e
	buildozer android clean
	buildozer -v android debug deploy run logcat

log:
	buildozer android logcat
