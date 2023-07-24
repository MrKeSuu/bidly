build-debug:
	LEGACY_NDK=~/.buildozer/android/platform/android-ndk-r21e buildozer -v android debug

deploy-log:
	buildozer android deploy run logcat

android:
	LEGACY_NDK=~/.buildozer/android/platform/android-ndk-r21e buildozer -v android debug deploy run

complete:
	buildozer android clean
	LEGACY_NDK=~/.buildozer/android/platform/android-ndk-r21e buildozer -v android debug deploy run logcat

log:
	buildozer android logcat
